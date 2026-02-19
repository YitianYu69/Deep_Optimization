class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 3, 7, 7]", primals_2: "f32[512, 3, 224, 224]", primals_3: "i64[]", primals_4: "f32[64]", primals_5: "f32[64]", primals_6: "f32[64]", primals_7: "f32[64]", primals_8: "f32[64, 64, 1, 1]", primals_9: "i64[]", primals_10: "f32[64]", primals_11: "f32[64]", primals_12: "f32[64]", primals_13: "f32[64]", primals_14: "f32[64, 64, 3, 3]", primals_15: "i64[]", primals_16: "f32[64]", primals_17: "f32[64]", primals_18: "f32[64]", primals_19: "f32[64]", primals_20: "f32[256, 64, 1, 1]", primals_21: "i64[]", primals_22: "f32[256]", primals_23: "f32[256]", primals_24: "f32[256]", primals_25: "f32[256]", primals_26: "f32[256, 64, 1, 1]", primals_27: "i64[]", primals_28: "f32[256]", primals_29: "f32[256]", primals_30: "f32[256]", primals_31: "f32[256]", primals_32: "f32[64, 256, 1, 1]", primals_33: "i64[]", primals_34: "f32[64]", primals_35: "f32[64]", primals_36: "f32[64]", primals_37: "f32[64]", primals_38: "f32[64, 64, 3, 3]", primals_39: "i64[]", primals_40: "f32[64]", primals_41: "f32[64]", primals_42: "f32[64]", primals_43: "f32[64]", primals_44: "f32[256, 64, 1, 1]", primals_45: "i64[]", primals_46: "f32[256]", primals_47: "f32[256]", primals_48: "f32[256]", primals_49: "f32[256]", primals_50: "f32[64, 256, 1, 1]", primals_51: "i64[]", primals_52: "f32[64]", primals_53: "f32[64]", primals_54: "f32[64]", primals_55: "f32[64]", primals_56: "f32[64, 64, 3, 3]", primals_57: "i64[]", primals_58: "f32[64]", primals_59: "f32[64]", primals_60: "f32[64]", primals_61: "f32[64]", primals_62: "f32[256, 64, 1, 1]", primals_63: "i64[]", primals_64: "f32[256]", primals_65: "f32[256]", primals_66: "f32[256]", primals_67: "f32[256]", primals_68: "f32[128, 256, 1, 1]", primals_69: "i64[]", primals_70: "f32[128]", primals_71: "f32[128]", primals_72: "f32[128]", primals_73: "f32[128]", primals_74: "f32[128, 128, 3, 3]", primals_75: "i64[]", primals_76: "f32[128]", primals_77: "f32[128]", primals_78: "f32[128]", primals_79: "f32[128]", primals_80: "f32[512, 128, 1, 1]", primals_81: "i64[]", primals_82: "f32[512]", primals_83: "f32[512]", primals_84: "f32[512]", primals_85: "f32[512]", primals_86: "f32[512, 256, 1, 1]", primals_87: "i64[]", primals_88: "f32[512]", primals_89: "f32[512]", primals_90: "f32[512]", primals_91: "f32[512]", primals_92: "f32[128, 512, 1, 1]", primals_93: "i64[]", primals_94: "f32[128]", primals_95: "f32[128]", primals_96: "f32[128]", primals_97: "f32[128]", primals_98: "f32[128, 128, 3, 3]", primals_99: "i64[]", primals_100: "f32[128]", primals_101: "f32[128]", primals_102: "f32[128]", primals_103: "f32[128]", primals_104: "f32[512, 128, 1, 1]", primals_105: "i64[]", primals_106: "f32[512]", primals_107: "f32[512]", primals_108: "f32[512]", primals_109: "f32[512]", primals_110: "f32[128, 512, 1, 1]", primals_111: "i64[]", primals_112: "f32[128]", primals_113: "f32[128]", primals_114: "f32[128]", primals_115: "f32[128]", primals_116: "f32[128, 128, 3, 3]", primals_117: "i64[]", primals_118: "f32[128]", primals_119: "f32[128]", primals_120: "f32[128]", primals_121: "f32[128]", primals_122: "f32[512, 128, 1, 1]", primals_123: "i64[]", primals_124: "f32[512]", primals_125: "f32[512]", primals_126: "f32[512]", primals_127: "f32[512]", primals_128: "f32[128, 512, 1, 1]", primals_129: "i64[]", primals_130: "f32[128]", primals_131: "f32[128]", primals_132: "f32[128]", primals_133: "f32[128]", primals_134: "f32[128, 128, 3, 3]", primals_135: "i64[]", primals_136: "f32[128]", primals_137: "f32[128]", primals_138: "f32[128]", primals_139: "f32[128]", primals_140: "f32[512, 128, 1, 1]", primals_141: "i64[]", primals_142: "f32[512]", primals_143: "f32[512]", primals_144: "f32[512]", primals_145: "f32[512]", primals_146: "f32[256, 512, 1, 1]", primals_147: "i64[]", primals_148: "f32[256]", primals_149: "f32[256]", primals_150: "f32[256]", primals_151: "f32[256]", primals_152: "f32[256, 256, 3, 3]", primals_153: "i64[]", primals_154: "f32[256]", primals_155: "f32[256]", primals_156: "f32[256]", primals_157: "f32[256]", primals_158: "f32[1024, 256, 1, 1]", primals_159: "i64[]", primals_160: "f32[1024]", primals_161: "f32[1024]", primals_162: "f32[1024]", primals_163: "f32[1024]", primals_164: "f32[1024, 512, 1, 1]", primals_165: "i64[]", primals_166: "f32[1024]", primals_167: "f32[1024]", primals_168: "f32[1024]", primals_169: "f32[1024]", primals_170: "f32[256, 1024, 1, 1]", primals_171: "i64[]", primals_172: "f32[256]", primals_173: "f32[256]", primals_174: "f32[256]", primals_175: "f32[256]", primals_176: "f32[256, 256, 3, 3]", primals_177: "i64[]", primals_178: "f32[256]", primals_179: "f32[256]", primals_180: "f32[256]", primals_181: "f32[256]", primals_182: "f32[1024, 256, 1, 1]", primals_183: "i64[]", primals_184: "f32[1024]", primals_185: "f32[1024]", primals_186: "f32[1024]", primals_187: "f32[1024]", primals_188: "f32[256, 1024, 1, 1]", primals_189: "i64[]", primals_190: "f32[256]", primals_191: "f32[256]", primals_192: "f32[256]", primals_193: "f32[256]", primals_194: "f32[256, 256, 3, 3]", primals_195: "i64[]", primals_196: "f32[256]", primals_197: "f32[256]", primals_198: "f32[256]", primals_199: "f32[256]", primals_200: "f32[1024, 256, 1, 1]", primals_201: "i64[]", primals_202: "f32[1024]", primals_203: "f32[1024]", primals_204: "f32[1024]", primals_205: "f32[1024]", primals_206: "f32[256, 1024, 1, 1]", primals_207: "i64[]", primals_208: "f32[256]", primals_209: "f32[256]", primals_210: "f32[256]", primals_211: "f32[256]", primals_212: "f32[256, 256, 3, 3]", primals_213: "i64[]", primals_214: "f32[256]", primals_215: "f32[256]", primals_216: "f32[256]", primals_217: "f32[256]", primals_218: "f32[1024, 256, 1, 1]", primals_219: "i64[]", primals_220: "f32[1024]", primals_221: "f32[1024]", primals_222: "f32[1024]", primals_223: "f32[1024]", primals_224: "f32[256, 1024, 1, 1]", primals_225: "i64[]", primals_226: "f32[256]", primals_227: "f32[256]", primals_228: "f32[256]", primals_229: "f32[256]", primals_230: "f32[256, 256, 3, 3]", primals_231: "i64[]", primals_232: "f32[256]", primals_233: "f32[256]", primals_234: "f32[256]", primals_235: "f32[256]", primals_236: "f32[1024, 256, 1, 1]", primals_237: "i64[]", primals_238: "f32[1024]", primals_239: "f32[1024]", primals_240: "f32[1024]", primals_241: "f32[1024]", primals_242: "f32[256, 1024, 1, 1]", primals_243: "i64[]", primals_244: "f32[256]", primals_245: "f32[256]", primals_246: "f32[256]", primals_247: "f32[256]", primals_248: "f32[256, 256, 3, 3]", primals_249: "i64[]", primals_250: "f32[256]", primals_251: "f32[256]", primals_252: "f32[256]", primals_253: "f32[256]", primals_254: "f32[1024, 256, 1, 1]", primals_255: "i64[]", primals_256: "f32[1024]", primals_257: "f32[1024]", primals_258: "f32[1024]", primals_259: "f32[1024]", primals_260: "f32[512, 1024, 1, 1]", primals_261: "i64[]", primals_262: "f32[512]", primals_263: "f32[512]", primals_264: "f32[512]", primals_265: "f32[512]", primals_266: "f32[512, 512, 3, 3]", primals_267: "i64[]", primals_268: "f32[512]", primals_269: "f32[512]", primals_270: "f32[512]", primals_271: "f32[512]", primals_272: "f32[2048, 512, 1, 1]", primals_273: "i64[]", primals_274: "f32[2048]", primals_275: "f32[2048]", primals_276: "f32[2048]", primals_277: "f32[2048]", primals_278: "f32[2048, 1024, 1, 1]", primals_279: "i64[]", primals_280: "f32[2048]", primals_281: "f32[2048]", primals_282: "f32[2048]", primals_283: "f32[2048]", primals_284: "f32[512, 2048, 1, 1]", primals_285: "i64[]", primals_286: "f32[512]", primals_287: "f32[512]", primals_288: "f32[512]", primals_289: "f32[512]", primals_290: "f32[512, 512, 3, 3]", primals_291: "i64[]", primals_292: "f32[512]", primals_293: "f32[512]", primals_294: "f32[512]", primals_295: "f32[512]", primals_296: "f32[2048, 512, 1, 1]", primals_297: "i64[]", primals_298: "f32[2048]", primals_299: "f32[2048]", primals_300: "f32[2048]", primals_301: "f32[2048]", primals_302: "f32[512, 2048, 1, 1]", primals_303: "i64[]", primals_304: "f32[512]", primals_305: "f32[512]", primals_306: "f32[512]", primals_307: "f32[512]", primals_308: "f32[512, 512, 3, 3]", primals_309: "i64[]", primals_310: "f32[512]", primals_311: "f32[512]", primals_312: "f32[512]", primals_313: "f32[512]", primals_314: "f32[2048, 512, 1, 1]", primals_315: "i64[]", primals_316: "f32[2048]", primals_317: "f32[2048]", primals_318: "f32[2048]", primals_319: "f32[2048]", primals_320: "f32[100, 2048]"):
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type: "bf16[512, 3, 224, 224]" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convert_element_type_1: "bf16[64, 3, 7, 7]" = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16);  primals_1 = None
        view_2: "bf16[512, 588, 256]" = torch.ops.aten.view.default(convert_element_type, [512, -1, 256])
        view_3: "bf16[301056, 256]" = torch.ops.aten.view.default(view_2, [301056, 256]);  view_2 = None
        empty: "i32[301056, 16]" = torch.ops.aten.empty.memory_format([301056, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_1: "bf16[301056]" = torch.ops.aten.empty.memory_format([301056], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_2: "bf16[301056]" = torch.ops.aten.empty.memory_format([301056], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 218, constant_args_idx = 311, grid = [(301056, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_3, 'P_ptr': empty, 'S_ptr': empty_1, 'M_ptr': empty_2, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_3 = empty = empty_1 = empty_2 = None
        getitem: "i32[301056, 16]" = triton_kernel_wrapper_functional_proxy['P_ptr']
        getitem_1: "bf16[301056]" = triton_kernel_wrapper_functional_proxy['S_ptr']
        getitem_2: "bf16[301056]" = triton_kernel_wrapper_functional_proxy['M_ptr'];  triton_kernel_wrapper_functional_proxy = None
        convolution: "bf16[512, 64, 112, 112]" = torch.ops.aten.convolution.default(convert_element_type, convert_element_type_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1);  convert_element_type = convert_element_type_1 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add: "i64[]" = torch.ops.aten.add.Tensor(primals_3, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_9: "bf16[512, 64, 12544]" = torch.ops.aten.view.default(convolution, [512, 64, 12544]);  convolution = None
        full_default: "f32[64]" = torch.ops.aten.full.default([64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_1 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 219, constant_args_idx = 312, grid = [(64, 6272, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_9, 'SUM': full_default, 'SUMSQ': full_default, 'M': 6422528, 'HW': 12544, 'stride_n': 802816, 'stride_c': 12544, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_3: "f32[64]" = triton_kernel_wrapper_functional_proxy_1['SUM']
        getitem_4: "f32[64]" = triton_kernel_wrapper_functional_proxy_1['SUMSQ'];  triton_kernel_wrapper_functional_proxy_1 = None
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 6422528.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        div: "f32[64]" = torch.ops.aten.div.Tensor(getitem_3, full_default_2);  getitem_3 = None
        div_1: "f32[64]" = torch.ops.aten.div.Tensor(getitem_4, full_default_2);  getitem_4 = full_default_2 = None
        mul_1: "f32[64]" = torch.ops.aten.mul.Tensor(div, div)
        sub: "f32[64]" = torch.ops.aten.sub.Tensor(div_1, mul_1);  div_1 = mul_1 = None
        clamp_min: "f32[64]" = torch.ops.aten.clamp_min.default(sub, 0.0);  sub = None
        add_1: "f32[64]" = torch.ops.aten.add.Tensor(clamp_min, 1e-05)
        rsqrt: "f32[64]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        full_default_3: "f32[]" = torch.ops.aten.full.default([], 1.0000001192092896, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        mul_2: "f32[64]" = torch.ops.aten.mul.Tensor(clamp_min, full_default_3);  clamp_min = full_default_3 = None
        mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(primals_6, 0.9)
        mul_4: "f32[64]" = torch.ops.aten.mul.Tensor(div, 0.1)
        add_2: "f32[64]" = torch.ops.aten.add.Tensor(mul_3, mul_4);  mul_3 = mul_4 = None
        mul_5: "f32[64]" = torch.ops.aten.mul.Tensor(primals_7, 0.9)
        mul_6: "f32[64]" = torch.ops.aten.mul.Tensor(mul_2, 0.1);  mul_2 = None
        add_3: "f32[64]" = torch.ops.aten.add.Tensor(mul_5, mul_6);  mul_5 = mul_6 = None
        empty_3: "bf16[512, 64, 12544]" = torch.ops.aten.empty.memory_format([512, 64, 12544], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute: "bf16[512, 64, 12544]" = torch.ops.aten.permute.default(empty_3, [0, 1, 2]);  empty_3 = None
        empty_4: "bf16[512, 64, 12544]" = torch.ops.aten.empty.memory_format([512, 64, 12544], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_1: "bf16[512, 64, 12544]" = torch.ops.aten.permute.default(empty_4, [0, 1, 2]);  empty_4 = None
        triton_kernel_wrapper_functional_proxy_2 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 220, constant_args_idx = 313, grid = [(64, 6272, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_9, 'MEAN': div, 'INVSTD': rsqrt, 'GAMMA': primals_4, 'BETA': primals_5, 'Y': permute, 'X_hat': permute_1, 'M': 6422528, 'HW': 12544, 'stride_n': 802816, 'stride_c': 12544, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_9 = div = primals_5 = permute = permute_1 = None
        getitem_5: "bf16[512, 64, 12544]" = triton_kernel_wrapper_functional_proxy_2['Y']
        getitem_6: "bf16[512, 64, 12544]" = triton_kernel_wrapper_functional_proxy_2['X_hat'];  triton_kernel_wrapper_functional_proxy_2 = None
        empty_5: "i32[1605632, 16]" = torch.ops.aten.empty.memory_format([1605632, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_6: "bf16[1605632]" = torch.ops.aten.empty.memory_format([1605632], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_7: "bf16[1605632]" = torch.ops.aten.empty.memory_format([1605632], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_15: "bf16[512, 3136, 256]" = torch.ops.aten.view.default(getitem_6, [512, -1, 256]);  getitem_6 = None
        view_16: "bf16[1605632, 256]" = torch.ops.aten.view.default(view_15, [1605632, 256]);  view_15 = None
        triton_kernel_wrapper_functional_proxy_3 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 221, constant_args_idx = 314, grid = [(1605632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_16, 'P_ptr': empty_5, 'S_ptr': empty_6, 'M_ptr': empty_7, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_16 = empty_5 = empty_6 = empty_7 = None
        getitem_7: "i32[1605632, 16]" = triton_kernel_wrapper_functional_proxy_3['P_ptr']
        getitem_8: "bf16[1605632]" = triton_kernel_wrapper_functional_proxy_3['S_ptr']
        getitem_9: "bf16[1605632]" = triton_kernel_wrapper_functional_proxy_3['M_ptr'];  triton_kernel_wrapper_functional_proxy_3 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_4: "i8[512, 64, 112, 112]" = torch.ops.aten.full.default([512, 64, 112, 112], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_22: "bf16[512, 64, 112, 112]" = torch.ops.aten.view.default(getitem_5, [512, 64, 112, 112]);  getitem_5 = None
        empty_8: "bf16[512, 64, 112, 112]" = torch.ops.aten.empty.memory_format([512, 64, 112, 112], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_2: "bf16[512, 64, 112, 112]" = torch.ops.aten.permute.default(empty_8, [0, 1, 2, 3]);  empty_8 = None
        triton_kernel_wrapper_functional_proxy_4 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 315, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_22, 'Y_ptr': permute_2, 'Mask_prt': full_default_4, 'n_elts': 411041792, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_22 = permute_2 = full_default_4 = None
        getitem_10: "bf16[512, 64, 112, 112]" = triton_kernel_wrapper_functional_proxy_4['Y_ptr']
        getitem_11: "i8[512, 64, 112, 112]" = triton_kernel_wrapper_functional_proxy_4['Mask_prt'];  triton_kernel_wrapper_functional_proxy_4 = None
        full_default_5: "i32[1605632, 8]" = torch.ops.aten.full.default([1605632, 8], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_27: "i8[512, 3136, 256]" = torch.ops.aten.view.default(getitem_11, [512, -1, 256]);  getitem_11 = None
        view_28: "i8[1605632, 256]" = torch.ops.aten.view.default(view_27, [1605632, 256]);  view_27 = None
        triton_kernel_wrapper_functional_proxy_5 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 316, grid = [(1605632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_28, 'P_ptr': full_default_5, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_28 = None
        getitem_12: "i32[1605632, 8]" = triton_kernel_wrapper_functional_proxy_5['P_ptr'];  triton_kernel_wrapper_functional_proxy_5 = None
        
         # File: <eval_with_key>.3 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:8 in forward, code: maxpool = self.maxpool(relu);  relu = None
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(getitem_10, [3, 3], [2, 2], [1, 1], [1, 1], False)
        getitem_13: "bf16[512, 64, 56, 56]" = _low_memory_max_pool_with_offsets[0]
        getitem_14: "i8[512, 64, 56, 56]" = _low_memory_max_pool_with_offsets[1];  _low_memory_max_pool_with_offsets = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_2: "bf16[64, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        view_31: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_13, [512, -1, 256])
        view_32: "bf16[401408, 256]" = torch.ops.aten.view.default(view_31, [401408, 256]);  view_31 = None
        empty_9: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_10: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_11: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_6 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 222, constant_args_idx = 317, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_32, 'P_ptr': empty_9, 'S_ptr': empty_10, 'M_ptr': empty_11, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  empty_9 = empty_10 = empty_11 = None
        getitem_15: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_6['P_ptr']
        getitem_16: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_6['S_ptr']
        getitem_17: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_6['M_ptr'];  triton_kernel_wrapper_functional_proxy_6 = None
        convolution_1: "bf16[512, 64, 56, 56]" = torch.ops.aten.convolution.default(getitem_13, convert_element_type_2, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_4: "i64[]" = torch.ops.aten.add.Tensor(primals_9, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_38: "bf16[512, 64, 3136]" = torch.ops.aten.view.default(convolution_1, [512, 64, 3136]);  convolution_1 = None
        triton_kernel_wrapper_functional_proxy_7 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 223, constant_args_idx = 318, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_38, 'SUM': full_default, 'SUMSQ': full_default, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_18: "f32[64]" = triton_kernel_wrapper_functional_proxy_7['SUM']
        getitem_19: "f32[64]" = triton_kernel_wrapper_functional_proxy_7['SUMSQ'];  triton_kernel_wrapper_functional_proxy_7 = None
        full_default_8: "f32[]" = torch.ops.aten.full.default([], 1605632.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        div_3: "f32[64]" = torch.ops.aten.div.Tensor(getitem_18, full_default_8);  getitem_18 = None
        div_4: "f32[64]" = torch.ops.aten.div.Tensor(getitem_19, full_default_8);  getitem_19 = None
        mul_8: "f32[64]" = torch.ops.aten.mul.Tensor(div_3, div_3)
        sub_2: "f32[64]" = torch.ops.aten.sub.Tensor(div_4, mul_8);  div_4 = mul_8 = None
        clamp_min_2: "f32[64]" = torch.ops.aten.clamp_min.default(sub_2, 0.0);  sub_2 = None
        add_5: "f32[64]" = torch.ops.aten.add.Tensor(clamp_min_2, 1e-05)
        rsqrt_1: "f32[64]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        full_default_9: "f32[]" = torch.ops.aten.full.default([], 1.0000005960464478, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        mul_9: "f32[64]" = torch.ops.aten.mul.Tensor(clamp_min_2, full_default_9);  clamp_min_2 = None
        mul_10: "f32[64]" = torch.ops.aten.mul.Tensor(primals_12, 0.9)
        mul_11: "f32[64]" = torch.ops.aten.mul.Tensor(div_3, 0.1)
        add_6: "f32[64]" = torch.ops.aten.add.Tensor(mul_10, mul_11);  mul_10 = mul_11 = None
        mul_12: "f32[64]" = torch.ops.aten.mul.Tensor(primals_13, 0.9)
        mul_13: "f32[64]" = torch.ops.aten.mul.Tensor(mul_9, 0.1);  mul_9 = None
        add_7: "f32[64]" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
        empty_12: "bf16[512, 64, 3136]" = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_3: "bf16[512, 64, 3136]" = torch.ops.aten.permute.default(empty_12, [0, 1, 2]);  empty_12 = None
        empty_13: "bf16[512, 64, 3136]" = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_4: "bf16[512, 64, 3136]" = torch.ops.aten.permute.default(empty_13, [0, 1, 2]);  empty_13 = None
        triton_kernel_wrapper_functional_proxy_8 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 224, constant_args_idx = 319, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_38, 'MEAN': div_3, 'INVSTD': rsqrt_1, 'GAMMA': primals_10, 'BETA': primals_11, 'Y': permute_3, 'X_hat': permute_4, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_38 = div_3 = primals_11 = permute_3 = permute_4 = None
        getitem_20: "bf16[512, 64, 3136]" = triton_kernel_wrapper_functional_proxy_8['Y']
        getitem_21: "bf16[512, 64, 3136]" = triton_kernel_wrapper_functional_proxy_8['X_hat'];  triton_kernel_wrapper_functional_proxy_8 = None
        empty_14: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_15: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_16: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_44: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_21, [512, -1, 256]);  getitem_21 = None
        view_45: "bf16[401408, 256]" = torch.ops.aten.view.default(view_44, [401408, 256]);  view_44 = None
        triton_kernel_wrapper_functional_proxy_9 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 225, constant_args_idx = 320, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_45, 'P_ptr': empty_14, 'S_ptr': empty_15, 'M_ptr': empty_16, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_45 = empty_14 = empty_15 = empty_16 = None
        getitem_22: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_9['P_ptr']
        getitem_23: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_9['S_ptr']
        getitem_24: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_9['M_ptr'];  triton_kernel_wrapper_functional_proxy_9 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_10: "i8[512, 64, 56, 56]" = torch.ops.aten.full.default([512, 64, 56, 56], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_51: "bf16[512, 64, 56, 56]" = torch.ops.aten.view.default(getitem_20, [512, 64, 56, 56]);  getitem_20 = None
        empty_17: "bf16[512, 64, 56, 56]" = torch.ops.aten.empty.memory_format([512, 64, 56, 56], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_5: "bf16[512, 64, 56, 56]" = torch.ops.aten.permute.default(empty_17, [0, 1, 2, 3]);  empty_17 = None
        triton_kernel_wrapper_functional_proxy_10 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 321, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_51, 'Y_ptr': permute_5, 'Mask_prt': full_default_10, 'n_elts': 102760448, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_51 = permute_5 = None
        getitem_25: "bf16[512, 64, 56, 56]" = triton_kernel_wrapper_functional_proxy_10['Y_ptr']
        getitem_26: "i8[512, 64, 56, 56]" = triton_kernel_wrapper_functional_proxy_10['Mask_prt'];  triton_kernel_wrapper_functional_proxy_10 = None
        full_default_11: "i32[401408, 8]" = torch.ops.aten.full.default([401408, 8], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_56: "i8[512, 784, 256]" = torch.ops.aten.view.default(getitem_26, [512, -1, 256]);  getitem_26 = None
        view_57: "i8[401408, 256]" = torch.ops.aten.view.default(view_56, [401408, 256]);  view_56 = None
        triton_kernel_wrapper_functional_proxy_11 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 322, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_57, 'P_ptr': full_default_11, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_57 = None
        getitem_27: "i32[401408, 8]" = triton_kernel_wrapper_functional_proxy_11['P_ptr'];  triton_kernel_wrapper_functional_proxy_11 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_3: "bf16[64, 64, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        empty_18: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_19: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_20: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_62: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_25, [512, -1, 256])
        view_63: "bf16[401408, 256]" = torch.ops.aten.view.default(view_62, [401408, 256]);  view_62 = None
        triton_kernel_wrapper_functional_proxy_12 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 226, constant_args_idx = 323, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_63, 'P_ptr': empty_18, 'S_ptr': empty_19, 'M_ptr': empty_20, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_63 = empty_18 = empty_19 = empty_20 = None
        getitem_28: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_12['P_ptr']
        getitem_29: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_12['S_ptr']
        getitem_30: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_12['M_ptr'];  triton_kernel_wrapper_functional_proxy_12 = None
        convolution_2: "bf16[512, 64, 56, 56]" = torch.ops.aten.convolution.default(getitem_25, convert_element_type_3, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_25 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_8: "i64[]" = torch.ops.aten.add.Tensor(primals_15, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_69: "bf16[512, 64, 3136]" = torch.ops.aten.view.default(convolution_2, [512, 64, 3136]);  convolution_2 = None
        triton_kernel_wrapper_functional_proxy_13 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 227, constant_args_idx = 324, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_69, 'SUM': full_default, 'SUMSQ': full_default, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_31: "f32[64]" = triton_kernel_wrapper_functional_proxy_13['SUM']
        getitem_32: "f32[64]" = triton_kernel_wrapper_functional_proxy_13['SUMSQ'];  triton_kernel_wrapper_functional_proxy_13 = None
        div_6: "f32[64]" = torch.ops.aten.div.Tensor(getitem_31, full_default_8);  getitem_31 = None
        div_7: "f32[64]" = torch.ops.aten.div.Tensor(getitem_32, full_default_8);  getitem_32 = None
        mul_15: "f32[64]" = torch.ops.aten.mul.Tensor(div_6, div_6)
        sub_4: "f32[64]" = torch.ops.aten.sub.Tensor(div_7, mul_15);  div_7 = mul_15 = None
        clamp_min_4: "f32[64]" = torch.ops.aten.clamp_min.default(sub_4, 0.0);  sub_4 = None
        add_9: "f32[64]" = torch.ops.aten.add.Tensor(clamp_min_4, 1e-05)
        rsqrt_2: "f32[64]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        mul_16: "f32[64]" = torch.ops.aten.mul.Tensor(clamp_min_4, full_default_9);  clamp_min_4 = None
        mul_17: "f32[64]" = torch.ops.aten.mul.Tensor(primals_18, 0.9)
        mul_18: "f32[64]" = torch.ops.aten.mul.Tensor(div_6, 0.1)
        add_10: "f32[64]" = torch.ops.aten.add.Tensor(mul_17, mul_18);  mul_17 = mul_18 = None
        mul_19: "f32[64]" = torch.ops.aten.mul.Tensor(primals_19, 0.9)
        mul_20: "f32[64]" = torch.ops.aten.mul.Tensor(mul_16, 0.1);  mul_16 = None
        add_11: "f32[64]" = torch.ops.aten.add.Tensor(mul_19, mul_20);  mul_19 = mul_20 = None
        empty_21: "bf16[512, 64, 3136]" = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_6: "bf16[512, 64, 3136]" = torch.ops.aten.permute.default(empty_21, [0, 1, 2]);  empty_21 = None
        empty_22: "bf16[512, 64, 3136]" = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_7: "bf16[512, 64, 3136]" = torch.ops.aten.permute.default(empty_22, [0, 1, 2]);  empty_22 = None
        triton_kernel_wrapper_functional_proxy_14 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 228, constant_args_idx = 325, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_69, 'MEAN': div_6, 'INVSTD': rsqrt_2, 'GAMMA': primals_16, 'BETA': primals_17, 'Y': permute_6, 'X_hat': permute_7, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_69 = div_6 = primals_17 = permute_6 = permute_7 = None
        getitem_33: "bf16[512, 64, 3136]" = triton_kernel_wrapper_functional_proxy_14['Y']
        getitem_34: "bf16[512, 64, 3136]" = triton_kernel_wrapper_functional_proxy_14['X_hat'];  triton_kernel_wrapper_functional_proxy_14 = None
        empty_23: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_24: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_25: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_75: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_34, [512, -1, 256]);  getitem_34 = None
        view_76: "bf16[401408, 256]" = torch.ops.aten.view.default(view_75, [401408, 256]);  view_75 = None
        triton_kernel_wrapper_functional_proxy_15 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 229, constant_args_idx = 326, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_76, 'P_ptr': empty_23, 'S_ptr': empty_24, 'M_ptr': empty_25, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_76 = empty_23 = empty_24 = empty_25 = None
        getitem_35: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_15['P_ptr']
        getitem_36: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_15['S_ptr']
        getitem_37: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_15['M_ptr'];  triton_kernel_wrapper_functional_proxy_15 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_82: "bf16[512, 64, 56, 56]" = torch.ops.aten.view.default(getitem_33, [512, 64, 56, 56]);  getitem_33 = None
        empty_26: "bf16[512, 64, 56, 56]" = torch.ops.aten.empty.memory_format([512, 64, 56, 56], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_8: "bf16[512, 64, 56, 56]" = torch.ops.aten.permute.default(empty_26, [0, 1, 2, 3]);  empty_26 = None
        triton_kernel_wrapper_functional_proxy_16 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 327, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_82, 'Y_ptr': permute_8, 'Mask_prt': full_default_10, 'n_elts': 102760448, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_82 = permute_8 = None
        getitem_38: "bf16[512, 64, 56, 56]" = triton_kernel_wrapper_functional_proxy_16['Y_ptr']
        getitem_39: "i8[512, 64, 56, 56]" = triton_kernel_wrapper_functional_proxy_16['Mask_prt'];  triton_kernel_wrapper_functional_proxy_16 = None
        view_87: "i8[512, 784, 256]" = torch.ops.aten.view.default(getitem_39, [512, -1, 256]);  getitem_39 = None
        view_88: "i8[401408, 256]" = torch.ops.aten.view.default(view_87, [401408, 256]);  view_87 = None
        triton_kernel_wrapper_functional_proxy_17 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 328, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_88, 'P_ptr': full_default_11, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_88 = None
        getitem_40: "i32[401408, 8]" = triton_kernel_wrapper_functional_proxy_17['P_ptr'];  triton_kernel_wrapper_functional_proxy_17 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_4: "bf16[256, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        empty_27: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_28: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_29: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_93: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_38, [512, -1, 256])
        view_94: "bf16[401408, 256]" = torch.ops.aten.view.default(view_93, [401408, 256]);  view_93 = None
        triton_kernel_wrapper_functional_proxy_18 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 230, constant_args_idx = 329, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_94, 'P_ptr': empty_27, 'S_ptr': empty_28, 'M_ptr': empty_29, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_94 = empty_27 = empty_28 = empty_29 = None
        getitem_41: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_18['P_ptr']
        getitem_42: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_18['S_ptr']
        getitem_43: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_18['M_ptr'];  triton_kernel_wrapper_functional_proxy_18 = None
        convolution_3: "bf16[512, 256, 56, 56]" = torch.ops.aten.convolution.default(getitem_38, convert_element_type_4, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_38 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_12: "i64[]" = torch.ops.aten.add.Tensor(primals_21, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_100: "bf16[512, 256, 3136]" = torch.ops.aten.view.default(convolution_3, [512, 256, 3136]);  convolution_3 = None
        full_default_18: "f32[256]" = torch.ops.aten.full.default([256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_19 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 231, constant_args_idx = 330, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_100, 'SUM': full_default_18, 'SUMSQ': full_default_18, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_44: "f32[256]" = triton_kernel_wrapper_functional_proxy_19['SUM']
        getitem_45: "f32[256]" = triton_kernel_wrapper_functional_proxy_19['SUMSQ'];  triton_kernel_wrapper_functional_proxy_19 = None
        div_9: "f32[256]" = torch.ops.aten.div.Tensor(getitem_44, full_default_8);  getitem_44 = None
        div_10: "f32[256]" = torch.ops.aten.div.Tensor(getitem_45, full_default_8);  getitem_45 = None
        mul_22: "f32[256]" = torch.ops.aten.mul.Tensor(div_9, div_9)
        sub_6: "f32[256]" = torch.ops.aten.sub.Tensor(div_10, mul_22);  div_10 = mul_22 = None
        clamp_min_6: "f32[256]" = torch.ops.aten.clamp_min.default(sub_6, 0.0);  sub_6 = None
        add_13: "f32[256]" = torch.ops.aten.add.Tensor(clamp_min_6, 1e-05)
        rsqrt_3: "f32[256]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        mul_23: "f32[256]" = torch.ops.aten.mul.Tensor(clamp_min_6, full_default_9);  clamp_min_6 = None
        mul_24: "f32[256]" = torch.ops.aten.mul.Tensor(primals_24, 0.9)
        mul_25: "f32[256]" = torch.ops.aten.mul.Tensor(div_9, 0.1)
        add_14: "f32[256]" = torch.ops.aten.add.Tensor(mul_24, mul_25);  mul_24 = mul_25 = None
        mul_26: "f32[256]" = torch.ops.aten.mul.Tensor(primals_25, 0.9)
        mul_27: "f32[256]" = torch.ops.aten.mul.Tensor(mul_23, 0.1);  mul_23 = None
        add_15: "f32[256]" = torch.ops.aten.add.Tensor(mul_26, mul_27);  mul_26 = mul_27 = None
        empty_30: "bf16[512, 256, 3136]" = torch.ops.aten.empty.memory_format([512, 256, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_9: "bf16[512, 256, 3136]" = torch.ops.aten.permute.default(empty_30, [0, 1, 2]);  empty_30 = None
        empty_31: "bf16[512, 256, 3136]" = torch.ops.aten.empty.memory_format([512, 256, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_10: "bf16[512, 256, 3136]" = torch.ops.aten.permute.default(empty_31, [0, 1, 2]);  empty_31 = None
        triton_kernel_wrapper_functional_proxy_20 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 232, constant_args_idx = 331, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_100, 'MEAN': div_9, 'INVSTD': rsqrt_3, 'GAMMA': primals_22, 'BETA': primals_23, 'Y': permute_9, 'X_hat': permute_10, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_100 = div_9 = primals_23 = permute_9 = permute_10 = None
        getitem_46: "bf16[512, 256, 3136]" = triton_kernel_wrapper_functional_proxy_20['Y']
        getitem_47: "bf16[512, 256, 3136]" = triton_kernel_wrapper_functional_proxy_20['X_hat'];  triton_kernel_wrapper_functional_proxy_20 = None
        empty_32: "i32[1605632, 16]" = torch.ops.aten.empty.memory_format([1605632, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_33: "bf16[1605632]" = torch.ops.aten.empty.memory_format([1605632], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_34: "bf16[1605632]" = torch.ops.aten.empty.memory_format([1605632], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_106: "bf16[512, 3136, 256]" = torch.ops.aten.view.default(getitem_47, [512, -1, 256]);  getitem_47 = None
        view_107: "bf16[1605632, 256]" = torch.ops.aten.view.default(view_106, [1605632, 256]);  view_106 = None
        triton_kernel_wrapper_functional_proxy_21 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 233, constant_args_idx = 332, grid = [(1605632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_107, 'P_ptr': empty_32, 'S_ptr': empty_33, 'M_ptr': empty_34, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_107 = empty_32 = empty_33 = empty_34 = None
        getitem_48: "i32[1605632, 16]" = triton_kernel_wrapper_functional_proxy_21['P_ptr']
        getitem_49: "bf16[1605632]" = triton_kernel_wrapper_functional_proxy_21['S_ptr']
        getitem_50: "bf16[1605632]" = triton_kernel_wrapper_functional_proxy_21['M_ptr'];  triton_kernel_wrapper_functional_proxy_21 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_5: "bf16[256, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16);  primals_26 = None
        empty_35: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_36: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_37: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_22 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 234, constant_args_idx = 333, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_32, 'P_ptr': empty_35, 'S_ptr': empty_36, 'M_ptr': empty_37, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_32 = empty_35 = empty_36 = empty_37 = None
        getitem_51: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_22['P_ptr']
        getitem_52: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_22['S_ptr']
        getitem_53: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_22['M_ptr'];  triton_kernel_wrapper_functional_proxy_22 = None
        convolution_4: "bf16[512, 256, 56, 56]" = torch.ops.aten.convolution.default(getitem_13, convert_element_type_5, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_13 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_16: "i64[]" = torch.ops.aten.add.Tensor(primals_27, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_122: "bf16[512, 256, 3136]" = torch.ops.aten.view.default(convolution_4, [512, 256, 3136]);  convolution_4 = None
        triton_kernel_wrapper_functional_proxy_23 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 235, constant_args_idx = 334, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_122, 'SUM': full_default_18, 'SUMSQ': full_default_18, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_54: "f32[256]" = triton_kernel_wrapper_functional_proxy_23['SUM']
        getitem_55: "f32[256]" = triton_kernel_wrapper_functional_proxy_23['SUMSQ'];  triton_kernel_wrapper_functional_proxy_23 = None
        div_12: "f32[256]" = torch.ops.aten.div.Tensor(getitem_54, full_default_8);  getitem_54 = None
        div_13: "f32[256]" = torch.ops.aten.div.Tensor(getitem_55, full_default_8);  getitem_55 = None
        mul_29: "f32[256]" = torch.ops.aten.mul.Tensor(div_12, div_12)
        sub_8: "f32[256]" = torch.ops.aten.sub.Tensor(div_13, mul_29);  div_13 = mul_29 = None
        clamp_min_8: "f32[256]" = torch.ops.aten.clamp_min.default(sub_8, 0.0);  sub_8 = None
        add_17: "f32[256]" = torch.ops.aten.add.Tensor(clamp_min_8, 1e-05)
        rsqrt_4: "f32[256]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_30: "f32[256]" = torch.ops.aten.mul.Tensor(clamp_min_8, full_default_9);  clamp_min_8 = None
        mul_31: "f32[256]" = torch.ops.aten.mul.Tensor(primals_30, 0.9)
        mul_32: "f32[256]" = torch.ops.aten.mul.Tensor(div_12, 0.1)
        add_18: "f32[256]" = torch.ops.aten.add.Tensor(mul_31, mul_32);  mul_31 = mul_32 = None
        mul_33: "f32[256]" = torch.ops.aten.mul.Tensor(primals_31, 0.9)
        mul_34: "f32[256]" = torch.ops.aten.mul.Tensor(mul_30, 0.1);  mul_30 = None
        add_19: "f32[256]" = torch.ops.aten.add.Tensor(mul_33, mul_34);  mul_33 = mul_34 = None
        empty_38: "bf16[512, 256, 3136]" = torch.ops.aten.empty.memory_format([512, 256, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_11: "bf16[512, 256, 3136]" = torch.ops.aten.permute.default(empty_38, [0, 1, 2]);  empty_38 = None
        empty_39: "bf16[512, 256, 3136]" = torch.ops.aten.empty.memory_format([512, 256, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_12: "bf16[512, 256, 3136]" = torch.ops.aten.permute.default(empty_39, [0, 1, 2]);  empty_39 = None
        triton_kernel_wrapper_functional_proxy_24 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 236, constant_args_idx = 335, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_122, 'MEAN': div_12, 'INVSTD': rsqrt_4, 'GAMMA': primals_28, 'BETA': primals_29, 'Y': permute_11, 'X_hat': permute_12, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_122 = div_12 = primals_29 = permute_11 = permute_12 = None
        getitem_56: "bf16[512, 256, 3136]" = triton_kernel_wrapper_functional_proxy_24['Y']
        getitem_57: "bf16[512, 256, 3136]" = triton_kernel_wrapper_functional_proxy_24['X_hat'];  triton_kernel_wrapper_functional_proxy_24 = None
        empty_40: "i32[1605632, 16]" = torch.ops.aten.empty.memory_format([1605632, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_41: "bf16[1605632]" = torch.ops.aten.empty.memory_format([1605632], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_42: "bf16[1605632]" = torch.ops.aten.empty.memory_format([1605632], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_128: "bf16[512, 3136, 256]" = torch.ops.aten.view.default(getitem_57, [512, -1, 256]);  getitem_57 = None
        view_129: "bf16[1605632, 256]" = torch.ops.aten.view.default(view_128, [1605632, 256]);  view_128 = None
        triton_kernel_wrapper_functional_proxy_25 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 237, constant_args_idx = 336, grid = [(1605632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_129, 'P_ptr': empty_40, 'S_ptr': empty_41, 'M_ptr': empty_42, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_129 = empty_40 = empty_41 = empty_42 = None
        getitem_58: "i32[1605632, 16]" = triton_kernel_wrapper_functional_proxy_25['P_ptr']
        getitem_59: "bf16[1605632]" = triton_kernel_wrapper_functional_proxy_25['S_ptr']
        getitem_60: "bf16[1605632]" = triton_kernel_wrapper_functional_proxy_25['M_ptr'];  triton_kernel_wrapper_functional_proxy_25 = None
        
         # File: <eval_with_key>.3 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:19 in forward, code: add = layer1_0_bn3 + layer1_0_downsample_1;  layer1_0_bn3 = layer1_0_downsample_1 = None
        view_135: "bf16[512, 256, 56, 56]" = torch.ops.aten.view.default(getitem_46, [512, 256, 56, 56]);  getitem_46 = None
        view_136: "bf16[512, 256, 56, 56]" = torch.ops.aten.view.default(getitem_56, [512, 256, 56, 56]);  getitem_56 = None
        add_20: "bf16[512, 256, 56, 56]" = torch.ops.aten.add.Tensor(view_135, view_136);  view_135 = view_136 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_26: "i8[512, 256, 56, 56]" = torch.ops.aten.full.default([512, 256, 56, 56], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        empty_43: "bf16[512, 256, 56, 56]" = torch.ops.aten.empty.memory_format([512, 256, 56, 56], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_13: "bf16[512, 256, 56, 56]" = torch.ops.aten.permute.default(empty_43, [0, 1, 2, 3]);  empty_43 = None
        triton_kernel_wrapper_functional_proxy_26 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 337, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_20, 'Y_ptr': permute_13, 'Mask_prt': full_default_26, 'n_elts': 411041792, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  add_20 = permute_13 = None
        getitem_61: "bf16[512, 256, 56, 56]" = triton_kernel_wrapper_functional_proxy_26['Y_ptr']
        getitem_62: "i8[512, 256, 56, 56]" = triton_kernel_wrapper_functional_proxy_26['Mask_prt'];  triton_kernel_wrapper_functional_proxy_26 = None
        view_141: "i8[512, 3136, 256]" = torch.ops.aten.view.default(getitem_62, [512, -1, 256]);  getitem_62 = None
        view_142: "i8[1605632, 256]" = torch.ops.aten.view.default(view_141, [1605632, 256]);  view_141 = None
        triton_kernel_wrapper_functional_proxy_27 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 338, grid = [(1605632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_142, 'P_ptr': full_default_5, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_142 = None
        getitem_63: "i32[1605632, 8]" = triton_kernel_wrapper_functional_proxy_27['P_ptr'];  triton_kernel_wrapper_functional_proxy_27 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_6: "bf16[64, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        empty_44: "i32[1605632, 16]" = torch.ops.aten.empty.memory_format([1605632, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_45: "bf16[1605632]" = torch.ops.aten.empty.memory_format([1605632], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_46: "bf16[1605632]" = torch.ops.aten.empty.memory_format([1605632], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_147: "bf16[512, 3136, 256]" = torch.ops.aten.view.default(getitem_61, [512, -1, 256])
        view_148: "bf16[1605632, 256]" = torch.ops.aten.view.default(view_147, [1605632, 256]);  view_147 = None
        triton_kernel_wrapper_functional_proxy_28 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 238, constant_args_idx = 339, grid = [(1605632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_148, 'P_ptr': empty_44, 'S_ptr': empty_45, 'M_ptr': empty_46, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_148 = empty_44 = empty_45 = empty_46 = None
        getitem_64: "i32[1605632, 16]" = triton_kernel_wrapper_functional_proxy_28['P_ptr']
        getitem_65: "bf16[1605632]" = triton_kernel_wrapper_functional_proxy_28['S_ptr']
        getitem_66: "bf16[1605632]" = triton_kernel_wrapper_functional_proxy_28['M_ptr'];  triton_kernel_wrapper_functional_proxy_28 = None
        convolution_5: "bf16[512, 64, 56, 56]" = torch.ops.aten.convolution.default(getitem_61, convert_element_type_6, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_21: "i64[]" = torch.ops.aten.add.Tensor(primals_33, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_154: "bf16[512, 64, 3136]" = torch.ops.aten.view.default(convolution_5, [512, 64, 3136]);  convolution_5 = None
        triton_kernel_wrapper_functional_proxy_29 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 239, constant_args_idx = 340, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_154, 'SUM': full_default, 'SUMSQ': full_default, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_67: "f32[64]" = triton_kernel_wrapper_functional_proxy_29['SUM']
        getitem_68: "f32[64]" = triton_kernel_wrapper_functional_proxy_29['SUMSQ'];  triton_kernel_wrapper_functional_proxy_29 = None
        div_15: "f32[64]" = torch.ops.aten.div.Tensor(getitem_67, full_default_8);  getitem_67 = None
        div_16: "f32[64]" = torch.ops.aten.div.Tensor(getitem_68, full_default_8);  getitem_68 = None
        mul_36: "f32[64]" = torch.ops.aten.mul.Tensor(div_15, div_15)
        sub_10: "f32[64]" = torch.ops.aten.sub.Tensor(div_16, mul_36);  div_16 = mul_36 = None
        clamp_min_10: "f32[64]" = torch.ops.aten.clamp_min.default(sub_10, 0.0);  sub_10 = None
        add_22: "f32[64]" = torch.ops.aten.add.Tensor(clamp_min_10, 1e-05)
        rsqrt_5: "f32[64]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_37: "f32[64]" = torch.ops.aten.mul.Tensor(clamp_min_10, full_default_9);  clamp_min_10 = None
        mul_38: "f32[64]" = torch.ops.aten.mul.Tensor(primals_36, 0.9)
        mul_39: "f32[64]" = torch.ops.aten.mul.Tensor(div_15, 0.1)
        add_23: "f32[64]" = torch.ops.aten.add.Tensor(mul_38, mul_39);  mul_38 = mul_39 = None
        mul_40: "f32[64]" = torch.ops.aten.mul.Tensor(primals_37, 0.9)
        mul_41: "f32[64]" = torch.ops.aten.mul.Tensor(mul_37, 0.1);  mul_37 = None
        add_24: "f32[64]" = torch.ops.aten.add.Tensor(mul_40, mul_41);  mul_40 = mul_41 = None
        empty_47: "bf16[512, 64, 3136]" = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_14: "bf16[512, 64, 3136]" = torch.ops.aten.permute.default(empty_47, [0, 1, 2]);  empty_47 = None
        empty_48: "bf16[512, 64, 3136]" = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_15: "bf16[512, 64, 3136]" = torch.ops.aten.permute.default(empty_48, [0, 1, 2]);  empty_48 = None
        triton_kernel_wrapper_functional_proxy_30 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 240, constant_args_idx = 341, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_154, 'MEAN': div_15, 'INVSTD': rsqrt_5, 'GAMMA': primals_34, 'BETA': primals_35, 'Y': permute_14, 'X_hat': permute_15, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_154 = div_15 = primals_35 = permute_14 = permute_15 = None
        getitem_69: "bf16[512, 64, 3136]" = triton_kernel_wrapper_functional_proxy_30['Y']
        getitem_70: "bf16[512, 64, 3136]" = triton_kernel_wrapper_functional_proxy_30['X_hat'];  triton_kernel_wrapper_functional_proxy_30 = None
        empty_49: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_50: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_51: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_160: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_70, [512, -1, 256]);  getitem_70 = None
        view_161: "bf16[401408, 256]" = torch.ops.aten.view.default(view_160, [401408, 256]);  view_160 = None
        triton_kernel_wrapper_functional_proxy_31 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 241, constant_args_idx = 342, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_161, 'P_ptr': empty_49, 'S_ptr': empty_50, 'M_ptr': empty_51, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_161 = empty_49 = empty_50 = empty_51 = None
        getitem_71: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_31['P_ptr']
        getitem_72: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_31['S_ptr']
        getitem_73: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_31['M_ptr'];  triton_kernel_wrapper_functional_proxy_31 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_167: "bf16[512, 64, 56, 56]" = torch.ops.aten.view.default(getitem_69, [512, 64, 56, 56]);  getitem_69 = None
        empty_52: "bf16[512, 64, 56, 56]" = torch.ops.aten.empty.memory_format([512, 64, 56, 56], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_16: "bf16[512, 64, 56, 56]" = torch.ops.aten.permute.default(empty_52, [0, 1, 2, 3]);  empty_52 = None
        triton_kernel_wrapper_functional_proxy_32 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 343, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_167, 'Y_ptr': permute_16, 'Mask_prt': full_default_10, 'n_elts': 102760448, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_167 = permute_16 = None
        getitem_74: "bf16[512, 64, 56, 56]" = triton_kernel_wrapper_functional_proxy_32['Y_ptr']
        getitem_75: "i8[512, 64, 56, 56]" = triton_kernel_wrapper_functional_proxy_32['Mask_prt'];  triton_kernel_wrapper_functional_proxy_32 = None
        view_172: "i8[512, 784, 256]" = torch.ops.aten.view.default(getitem_75, [512, -1, 256]);  getitem_75 = None
        view_173: "i8[401408, 256]" = torch.ops.aten.view.default(view_172, [401408, 256]);  view_172 = None
        triton_kernel_wrapper_functional_proxy_33 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 344, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_173, 'P_ptr': full_default_11, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_173 = None
        getitem_76: "i32[401408, 8]" = triton_kernel_wrapper_functional_proxy_33['P_ptr'];  triton_kernel_wrapper_functional_proxy_33 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_7: "bf16[64, 64, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        empty_53: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_54: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_55: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_178: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_74, [512, -1, 256])
        view_179: "bf16[401408, 256]" = torch.ops.aten.view.default(view_178, [401408, 256]);  view_178 = None
        triton_kernel_wrapper_functional_proxy_34 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 242, constant_args_idx = 345, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_179, 'P_ptr': empty_53, 'S_ptr': empty_54, 'M_ptr': empty_55, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_179 = empty_53 = empty_54 = empty_55 = None
        getitem_77: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_34['P_ptr']
        getitem_78: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_34['S_ptr']
        getitem_79: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_34['M_ptr'];  triton_kernel_wrapper_functional_proxy_34 = None
        convolution_6: "bf16[512, 64, 56, 56]" = torch.ops.aten.convolution.default(getitem_74, convert_element_type_7, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_74 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_25: "i64[]" = torch.ops.aten.add.Tensor(primals_39, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_185: "bf16[512, 64, 3136]" = torch.ops.aten.view.default(convolution_6, [512, 64, 3136]);  convolution_6 = None
        triton_kernel_wrapper_functional_proxy_35 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 243, constant_args_idx = 346, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_185, 'SUM': full_default, 'SUMSQ': full_default, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_80: "f32[64]" = triton_kernel_wrapper_functional_proxy_35['SUM']
        getitem_81: "f32[64]" = triton_kernel_wrapper_functional_proxy_35['SUMSQ'];  triton_kernel_wrapper_functional_proxy_35 = None
        div_18: "f32[64]" = torch.ops.aten.div.Tensor(getitem_80, full_default_8);  getitem_80 = None
        div_19: "f32[64]" = torch.ops.aten.div.Tensor(getitem_81, full_default_8);  getitem_81 = None
        mul_43: "f32[64]" = torch.ops.aten.mul.Tensor(div_18, div_18)
        sub_12: "f32[64]" = torch.ops.aten.sub.Tensor(div_19, mul_43);  div_19 = mul_43 = None
        clamp_min_12: "f32[64]" = torch.ops.aten.clamp_min.default(sub_12, 0.0);  sub_12 = None
        add_26: "f32[64]" = torch.ops.aten.add.Tensor(clamp_min_12, 1e-05)
        rsqrt_6: "f32[64]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        mul_44: "f32[64]" = torch.ops.aten.mul.Tensor(clamp_min_12, full_default_9);  clamp_min_12 = None
        mul_45: "f32[64]" = torch.ops.aten.mul.Tensor(primals_42, 0.9)
        mul_46: "f32[64]" = torch.ops.aten.mul.Tensor(div_18, 0.1)
        add_27: "f32[64]" = torch.ops.aten.add.Tensor(mul_45, mul_46);  mul_45 = mul_46 = None
        mul_47: "f32[64]" = torch.ops.aten.mul.Tensor(primals_43, 0.9)
        mul_48: "f32[64]" = torch.ops.aten.mul.Tensor(mul_44, 0.1);  mul_44 = None
        add_28: "f32[64]" = torch.ops.aten.add.Tensor(mul_47, mul_48);  mul_47 = mul_48 = None
        empty_56: "bf16[512, 64, 3136]" = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_17: "bf16[512, 64, 3136]" = torch.ops.aten.permute.default(empty_56, [0, 1, 2]);  empty_56 = None
        empty_57: "bf16[512, 64, 3136]" = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_18: "bf16[512, 64, 3136]" = torch.ops.aten.permute.default(empty_57, [0, 1, 2]);  empty_57 = None
        triton_kernel_wrapper_functional_proxy_36 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 244, constant_args_idx = 347, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_185, 'MEAN': div_18, 'INVSTD': rsqrt_6, 'GAMMA': primals_40, 'BETA': primals_41, 'Y': permute_17, 'X_hat': permute_18, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_185 = div_18 = primals_41 = permute_17 = permute_18 = None
        getitem_82: "bf16[512, 64, 3136]" = triton_kernel_wrapper_functional_proxy_36['Y']
        getitem_83: "bf16[512, 64, 3136]" = triton_kernel_wrapper_functional_proxy_36['X_hat'];  triton_kernel_wrapper_functional_proxy_36 = None
        empty_58: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_59: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_60: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_191: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_83, [512, -1, 256]);  getitem_83 = None
        view_192: "bf16[401408, 256]" = torch.ops.aten.view.default(view_191, [401408, 256]);  view_191 = None
        triton_kernel_wrapper_functional_proxy_37 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 245, constant_args_idx = 348, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_192, 'P_ptr': empty_58, 'S_ptr': empty_59, 'M_ptr': empty_60, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_192 = empty_58 = empty_59 = empty_60 = None
        getitem_84: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_37['P_ptr']
        getitem_85: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_37['S_ptr']
        getitem_86: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_37['M_ptr'];  triton_kernel_wrapper_functional_proxy_37 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_198: "bf16[512, 64, 56, 56]" = torch.ops.aten.view.default(getitem_82, [512, 64, 56, 56]);  getitem_82 = None
        empty_61: "bf16[512, 64, 56, 56]" = torch.ops.aten.empty.memory_format([512, 64, 56, 56], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_19: "bf16[512, 64, 56, 56]" = torch.ops.aten.permute.default(empty_61, [0, 1, 2, 3]);  empty_61 = None
        triton_kernel_wrapper_functional_proxy_38 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 349, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_198, 'Y_ptr': permute_19, 'Mask_prt': full_default_10, 'n_elts': 102760448, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_198 = permute_19 = None
        getitem_87: "bf16[512, 64, 56, 56]" = triton_kernel_wrapper_functional_proxy_38['Y_ptr']
        getitem_88: "i8[512, 64, 56, 56]" = triton_kernel_wrapper_functional_proxy_38['Mask_prt'];  triton_kernel_wrapper_functional_proxy_38 = None
        view_203: "i8[512, 784, 256]" = torch.ops.aten.view.default(getitem_88, [512, -1, 256]);  getitem_88 = None
        view_204: "i8[401408, 256]" = torch.ops.aten.view.default(view_203, [401408, 256]);  view_203 = None
        triton_kernel_wrapper_functional_proxy_39 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 350, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_204, 'P_ptr': full_default_11, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_204 = None
        getitem_89: "i32[401408, 8]" = triton_kernel_wrapper_functional_proxy_39['P_ptr'];  triton_kernel_wrapper_functional_proxy_39 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_8: "bf16[256, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        empty_62: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_63: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_64: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_209: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_87, [512, -1, 256])
        view_210: "bf16[401408, 256]" = torch.ops.aten.view.default(view_209, [401408, 256]);  view_209 = None
        triton_kernel_wrapper_functional_proxy_40 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 246, constant_args_idx = 351, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_210, 'P_ptr': empty_62, 'S_ptr': empty_63, 'M_ptr': empty_64, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_210 = empty_62 = empty_63 = empty_64 = None
        getitem_90: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_40['P_ptr']
        getitem_91: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_40['S_ptr']
        getitem_92: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_40['M_ptr'];  triton_kernel_wrapper_functional_proxy_40 = None
        convolution_7: "bf16[512, 256, 56, 56]" = torch.ops.aten.convolution.default(getitem_87, convert_element_type_8, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_87 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_29: "i64[]" = torch.ops.aten.add.Tensor(primals_45, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_216: "bf16[512, 256, 3136]" = torch.ops.aten.view.default(convolution_7, [512, 256, 3136]);  convolution_7 = None
        triton_kernel_wrapper_functional_proxy_41 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 247, constant_args_idx = 352, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_216, 'SUM': full_default_18, 'SUMSQ': full_default_18, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_93: "f32[256]" = triton_kernel_wrapper_functional_proxy_41['SUM']
        getitem_94: "f32[256]" = triton_kernel_wrapper_functional_proxy_41['SUMSQ'];  triton_kernel_wrapper_functional_proxy_41 = None
        div_21: "f32[256]" = torch.ops.aten.div.Tensor(getitem_93, full_default_8);  getitem_93 = None
        div_22: "f32[256]" = torch.ops.aten.div.Tensor(getitem_94, full_default_8);  getitem_94 = None
        mul_50: "f32[256]" = torch.ops.aten.mul.Tensor(div_21, div_21)
        sub_14: "f32[256]" = torch.ops.aten.sub.Tensor(div_22, mul_50);  div_22 = mul_50 = None
        clamp_min_14: "f32[256]" = torch.ops.aten.clamp_min.default(sub_14, 0.0);  sub_14 = None
        add_30: "f32[256]" = torch.ops.aten.add.Tensor(clamp_min_14, 1e-05)
        rsqrt_7: "f32[256]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_51: "f32[256]" = torch.ops.aten.mul.Tensor(clamp_min_14, full_default_9);  clamp_min_14 = None
        mul_52: "f32[256]" = torch.ops.aten.mul.Tensor(primals_48, 0.9)
        mul_53: "f32[256]" = torch.ops.aten.mul.Tensor(div_21, 0.1)
        add_31: "f32[256]" = torch.ops.aten.add.Tensor(mul_52, mul_53);  mul_52 = mul_53 = None
        mul_54: "f32[256]" = torch.ops.aten.mul.Tensor(primals_49, 0.9)
        mul_55: "f32[256]" = torch.ops.aten.mul.Tensor(mul_51, 0.1);  mul_51 = None
        add_32: "f32[256]" = torch.ops.aten.add.Tensor(mul_54, mul_55);  mul_54 = mul_55 = None
        empty_65: "bf16[512, 256, 3136]" = torch.ops.aten.empty.memory_format([512, 256, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_20: "bf16[512, 256, 3136]" = torch.ops.aten.permute.default(empty_65, [0, 1, 2]);  empty_65 = None
        empty_66: "bf16[512, 256, 3136]" = torch.ops.aten.empty.memory_format([512, 256, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_21: "bf16[512, 256, 3136]" = torch.ops.aten.permute.default(empty_66, [0, 1, 2]);  empty_66 = None
        triton_kernel_wrapper_functional_proxy_42 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 248, constant_args_idx = 353, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_216, 'MEAN': div_21, 'INVSTD': rsqrt_7, 'GAMMA': primals_46, 'BETA': primals_47, 'Y': permute_20, 'X_hat': permute_21, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_216 = div_21 = primals_47 = permute_20 = permute_21 = None
        getitem_95: "bf16[512, 256, 3136]" = triton_kernel_wrapper_functional_proxy_42['Y']
        getitem_96: "bf16[512, 256, 3136]" = triton_kernel_wrapper_functional_proxy_42['X_hat'];  triton_kernel_wrapper_functional_proxy_42 = None
        empty_67: "i32[1605632, 16]" = torch.ops.aten.empty.memory_format([1605632, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_68: "bf16[1605632]" = torch.ops.aten.empty.memory_format([1605632], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_69: "bf16[1605632]" = torch.ops.aten.empty.memory_format([1605632], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_222: "bf16[512, 3136, 256]" = torch.ops.aten.view.default(getitem_96, [512, -1, 256]);  getitem_96 = None
        view_223: "bf16[1605632, 256]" = torch.ops.aten.view.default(view_222, [1605632, 256]);  view_222 = None
        triton_kernel_wrapper_functional_proxy_43 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 249, constant_args_idx = 354, grid = [(1605632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_223, 'P_ptr': empty_67, 'S_ptr': empty_68, 'M_ptr': empty_69, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_223 = empty_67 = empty_68 = empty_69 = None
        getitem_97: "i32[1605632, 16]" = triton_kernel_wrapper_functional_proxy_43['P_ptr']
        getitem_98: "bf16[1605632]" = triton_kernel_wrapper_functional_proxy_43['S_ptr']
        getitem_99: "bf16[1605632]" = triton_kernel_wrapper_functional_proxy_43['M_ptr'];  triton_kernel_wrapper_functional_proxy_43 = None
        
         # File: <eval_with_key>.3 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:29 in forward, code: add_1 = layer1_1_bn3 + layer1_0_relu_2;  layer1_1_bn3 = layer1_0_relu_2 = None
        view_229: "bf16[512, 256, 56, 56]" = torch.ops.aten.view.default(getitem_95, [512, 256, 56, 56]);  getitem_95 = None
        add_33: "bf16[512, 256, 56, 56]" = torch.ops.aten.add.Tensor(view_229, getitem_61);  view_229 = getitem_61 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        empty_70: "bf16[512, 256, 56, 56]" = torch.ops.aten.empty.memory_format([512, 256, 56, 56], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_22: "bf16[512, 256, 56, 56]" = torch.ops.aten.permute.default(empty_70, [0, 1, 2, 3]);  empty_70 = None
        triton_kernel_wrapper_functional_proxy_44 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 355, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_33, 'Y_ptr': permute_22, 'Mask_prt': full_default_26, 'n_elts': 411041792, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  add_33 = permute_22 = None
        getitem_100: "bf16[512, 256, 56, 56]" = triton_kernel_wrapper_functional_proxy_44['Y_ptr']
        getitem_101: "i8[512, 256, 56, 56]" = triton_kernel_wrapper_functional_proxy_44['Mask_prt'];  triton_kernel_wrapper_functional_proxy_44 = None
        view_234: "i8[512, 3136, 256]" = torch.ops.aten.view.default(getitem_101, [512, -1, 256]);  getitem_101 = None
        view_235: "i8[1605632, 256]" = torch.ops.aten.view.default(view_234, [1605632, 256]);  view_234 = None
        triton_kernel_wrapper_functional_proxy_45 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 356, grid = [(1605632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_235, 'P_ptr': full_default_5, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_235 = None
        getitem_102: "i32[1605632, 8]" = triton_kernel_wrapper_functional_proxy_45['P_ptr'];  triton_kernel_wrapper_functional_proxy_45 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_9: "bf16[64, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16);  primals_50 = None
        empty_71: "i32[1605632, 16]" = torch.ops.aten.empty.memory_format([1605632, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_72: "bf16[1605632]" = torch.ops.aten.empty.memory_format([1605632], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_73: "bf16[1605632]" = torch.ops.aten.empty.memory_format([1605632], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_240: "bf16[512, 3136, 256]" = torch.ops.aten.view.default(getitem_100, [512, -1, 256])
        view_241: "bf16[1605632, 256]" = torch.ops.aten.view.default(view_240, [1605632, 256]);  view_240 = None
        triton_kernel_wrapper_functional_proxy_46 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 250, constant_args_idx = 357, grid = [(1605632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_241, 'P_ptr': empty_71, 'S_ptr': empty_72, 'M_ptr': empty_73, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_241 = empty_71 = empty_72 = empty_73 = None
        getitem_103: "i32[1605632, 16]" = triton_kernel_wrapper_functional_proxy_46['P_ptr']
        getitem_104: "bf16[1605632]" = triton_kernel_wrapper_functional_proxy_46['S_ptr']
        getitem_105: "bf16[1605632]" = triton_kernel_wrapper_functional_proxy_46['M_ptr'];  triton_kernel_wrapper_functional_proxy_46 = None
        convolution_8: "bf16[512, 64, 56, 56]" = torch.ops.aten.convolution.default(getitem_100, convert_element_type_9, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_34: "i64[]" = torch.ops.aten.add.Tensor(primals_51, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_247: "bf16[512, 64, 3136]" = torch.ops.aten.view.default(convolution_8, [512, 64, 3136]);  convolution_8 = None
        triton_kernel_wrapper_functional_proxy_47 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 251, constant_args_idx = 358, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_247, 'SUM': full_default, 'SUMSQ': full_default, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_106: "f32[64]" = triton_kernel_wrapper_functional_proxy_47['SUM']
        getitem_107: "f32[64]" = triton_kernel_wrapper_functional_proxy_47['SUMSQ'];  triton_kernel_wrapper_functional_proxy_47 = None
        div_24: "f32[64]" = torch.ops.aten.div.Tensor(getitem_106, full_default_8);  getitem_106 = None
        div_25: "f32[64]" = torch.ops.aten.div.Tensor(getitem_107, full_default_8);  getitem_107 = None
        mul_57: "f32[64]" = torch.ops.aten.mul.Tensor(div_24, div_24)
        sub_16: "f32[64]" = torch.ops.aten.sub.Tensor(div_25, mul_57);  div_25 = mul_57 = None
        clamp_min_16: "f32[64]" = torch.ops.aten.clamp_min.default(sub_16, 0.0);  sub_16 = None
        add_35: "f32[64]" = torch.ops.aten.add.Tensor(clamp_min_16, 1e-05)
        rsqrt_8: "f32[64]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        mul_58: "f32[64]" = torch.ops.aten.mul.Tensor(clamp_min_16, full_default_9);  clamp_min_16 = None
        mul_59: "f32[64]" = torch.ops.aten.mul.Tensor(primals_54, 0.9)
        mul_60: "f32[64]" = torch.ops.aten.mul.Tensor(div_24, 0.1)
        add_36: "f32[64]" = torch.ops.aten.add.Tensor(mul_59, mul_60);  mul_59 = mul_60 = None
        mul_61: "f32[64]" = torch.ops.aten.mul.Tensor(primals_55, 0.9)
        mul_62: "f32[64]" = torch.ops.aten.mul.Tensor(mul_58, 0.1);  mul_58 = None
        add_37: "f32[64]" = torch.ops.aten.add.Tensor(mul_61, mul_62);  mul_61 = mul_62 = None
        empty_74: "bf16[512, 64, 3136]" = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_23: "bf16[512, 64, 3136]" = torch.ops.aten.permute.default(empty_74, [0, 1, 2]);  empty_74 = None
        empty_75: "bf16[512, 64, 3136]" = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_24: "bf16[512, 64, 3136]" = torch.ops.aten.permute.default(empty_75, [0, 1, 2]);  empty_75 = None
        triton_kernel_wrapper_functional_proxy_48 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 252, constant_args_idx = 359, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_247, 'MEAN': div_24, 'INVSTD': rsqrt_8, 'GAMMA': primals_52, 'BETA': primals_53, 'Y': permute_23, 'X_hat': permute_24, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_247 = div_24 = primals_53 = permute_23 = permute_24 = None
        getitem_108: "bf16[512, 64, 3136]" = triton_kernel_wrapper_functional_proxy_48['Y']
        getitem_109: "bf16[512, 64, 3136]" = triton_kernel_wrapper_functional_proxy_48['X_hat'];  triton_kernel_wrapper_functional_proxy_48 = None
        empty_76: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_77: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_78: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_253: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_109, [512, -1, 256]);  getitem_109 = None
        view_254: "bf16[401408, 256]" = torch.ops.aten.view.default(view_253, [401408, 256]);  view_253 = None
        triton_kernel_wrapper_functional_proxy_49 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 253, constant_args_idx = 360, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_254, 'P_ptr': empty_76, 'S_ptr': empty_77, 'M_ptr': empty_78, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_254 = empty_76 = empty_77 = empty_78 = None
        getitem_110: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_49['P_ptr']
        getitem_111: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_49['S_ptr']
        getitem_112: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_49['M_ptr'];  triton_kernel_wrapper_functional_proxy_49 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_260: "bf16[512, 64, 56, 56]" = torch.ops.aten.view.default(getitem_108, [512, 64, 56, 56]);  getitem_108 = None
        empty_79: "bf16[512, 64, 56, 56]" = torch.ops.aten.empty.memory_format([512, 64, 56, 56], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_25: "bf16[512, 64, 56, 56]" = torch.ops.aten.permute.default(empty_79, [0, 1, 2, 3]);  empty_79 = None
        triton_kernel_wrapper_functional_proxy_50 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 361, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_260, 'Y_ptr': permute_25, 'Mask_prt': full_default_10, 'n_elts': 102760448, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_260 = permute_25 = None
        getitem_113: "bf16[512, 64, 56, 56]" = triton_kernel_wrapper_functional_proxy_50['Y_ptr']
        getitem_114: "i8[512, 64, 56, 56]" = triton_kernel_wrapper_functional_proxy_50['Mask_prt'];  triton_kernel_wrapper_functional_proxy_50 = None
        view_265: "i8[512, 784, 256]" = torch.ops.aten.view.default(getitem_114, [512, -1, 256]);  getitem_114 = None
        view_266: "i8[401408, 256]" = torch.ops.aten.view.default(view_265, [401408, 256]);  view_265 = None
        triton_kernel_wrapper_functional_proxy_51 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 362, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_266, 'P_ptr': full_default_11, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_266 = None
        getitem_115: "i32[401408, 8]" = triton_kernel_wrapper_functional_proxy_51['P_ptr'];  triton_kernel_wrapper_functional_proxy_51 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_10: "bf16[64, 64, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_56, torch.bfloat16);  primals_56 = None
        empty_80: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_81: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_82: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_271: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_113, [512, -1, 256])
        view_272: "bf16[401408, 256]" = torch.ops.aten.view.default(view_271, [401408, 256]);  view_271 = None
        triton_kernel_wrapper_functional_proxy_52 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 254, constant_args_idx = 363, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_272, 'P_ptr': empty_80, 'S_ptr': empty_81, 'M_ptr': empty_82, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_272 = empty_80 = empty_81 = empty_82 = None
        getitem_116: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_52['P_ptr']
        getitem_117: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_52['S_ptr']
        getitem_118: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_52['M_ptr'];  triton_kernel_wrapper_functional_proxy_52 = None
        convolution_9: "bf16[512, 64, 56, 56]" = torch.ops.aten.convolution.default(getitem_113, convert_element_type_10, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_113 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_38: "i64[]" = torch.ops.aten.add.Tensor(primals_57, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_278: "bf16[512, 64, 3136]" = torch.ops.aten.view.default(convolution_9, [512, 64, 3136]);  convolution_9 = None
        triton_kernel_wrapper_functional_proxy_53 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 255, constant_args_idx = 364, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_278, 'SUM': full_default, 'SUMSQ': full_default, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ']);  full_default = None
        getitem_119: "f32[64]" = triton_kernel_wrapper_functional_proxy_53['SUM']
        getitem_120: "f32[64]" = triton_kernel_wrapper_functional_proxy_53['SUMSQ'];  triton_kernel_wrapper_functional_proxy_53 = None
        div_27: "f32[64]" = torch.ops.aten.div.Tensor(getitem_119, full_default_8);  getitem_119 = None
        div_28: "f32[64]" = torch.ops.aten.div.Tensor(getitem_120, full_default_8);  getitem_120 = None
        mul_64: "f32[64]" = torch.ops.aten.mul.Tensor(div_27, div_27)
        sub_18: "f32[64]" = torch.ops.aten.sub.Tensor(div_28, mul_64);  div_28 = mul_64 = None
        clamp_min_18: "f32[64]" = torch.ops.aten.clamp_min.default(sub_18, 0.0);  sub_18 = None
        add_39: "f32[64]" = torch.ops.aten.add.Tensor(clamp_min_18, 1e-05)
        rsqrt_9: "f32[64]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        mul_65: "f32[64]" = torch.ops.aten.mul.Tensor(clamp_min_18, full_default_9);  clamp_min_18 = None
        mul_66: "f32[64]" = torch.ops.aten.mul.Tensor(primals_60, 0.9)
        mul_67: "f32[64]" = torch.ops.aten.mul.Tensor(div_27, 0.1)
        add_40: "f32[64]" = torch.ops.aten.add.Tensor(mul_66, mul_67);  mul_66 = mul_67 = None
        mul_68: "f32[64]" = torch.ops.aten.mul.Tensor(primals_61, 0.9)
        mul_69: "f32[64]" = torch.ops.aten.mul.Tensor(mul_65, 0.1);  mul_65 = None
        add_41: "f32[64]" = torch.ops.aten.add.Tensor(mul_68, mul_69);  mul_68 = mul_69 = None
        empty_83: "bf16[512, 64, 3136]" = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_26: "bf16[512, 64, 3136]" = torch.ops.aten.permute.default(empty_83, [0, 1, 2]);  empty_83 = None
        empty_84: "bf16[512, 64, 3136]" = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_27: "bf16[512, 64, 3136]" = torch.ops.aten.permute.default(empty_84, [0, 1, 2]);  empty_84 = None
        triton_kernel_wrapper_functional_proxy_54 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 256, constant_args_idx = 365, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_278, 'MEAN': div_27, 'INVSTD': rsqrt_9, 'GAMMA': primals_58, 'BETA': primals_59, 'Y': permute_26, 'X_hat': permute_27, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_278 = div_27 = primals_59 = permute_26 = permute_27 = None
        getitem_121: "bf16[512, 64, 3136]" = triton_kernel_wrapper_functional_proxy_54['Y']
        getitem_122: "bf16[512, 64, 3136]" = triton_kernel_wrapper_functional_proxy_54['X_hat'];  triton_kernel_wrapper_functional_proxy_54 = None
        empty_85: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_86: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_87: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_284: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_122, [512, -1, 256]);  getitem_122 = None
        view_285: "bf16[401408, 256]" = torch.ops.aten.view.default(view_284, [401408, 256]);  view_284 = None
        triton_kernel_wrapper_functional_proxy_55 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 257, constant_args_idx = 366, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_285, 'P_ptr': empty_85, 'S_ptr': empty_86, 'M_ptr': empty_87, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_285 = empty_85 = empty_86 = empty_87 = None
        getitem_123: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_55['P_ptr']
        getitem_124: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_55['S_ptr']
        getitem_125: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_55['M_ptr'];  triton_kernel_wrapper_functional_proxy_55 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_291: "bf16[512, 64, 56, 56]" = torch.ops.aten.view.default(getitem_121, [512, 64, 56, 56]);  getitem_121 = None
        empty_88: "bf16[512, 64, 56, 56]" = torch.ops.aten.empty.memory_format([512, 64, 56, 56], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_28: "bf16[512, 64, 56, 56]" = torch.ops.aten.permute.default(empty_88, [0, 1, 2, 3]);  empty_88 = None
        triton_kernel_wrapper_functional_proxy_56 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 367, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_291, 'Y_ptr': permute_28, 'Mask_prt': full_default_10, 'n_elts': 102760448, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_291 = permute_28 = full_default_10 = None
        getitem_126: "bf16[512, 64, 56, 56]" = triton_kernel_wrapper_functional_proxy_56['Y_ptr']
        getitem_127: "i8[512, 64, 56, 56]" = triton_kernel_wrapper_functional_proxy_56['Mask_prt'];  triton_kernel_wrapper_functional_proxy_56 = None
        view_296: "i8[512, 784, 256]" = torch.ops.aten.view.default(getitem_127, [512, -1, 256]);  getitem_127 = None
        view_297: "i8[401408, 256]" = torch.ops.aten.view.default(view_296, [401408, 256]);  view_296 = None
        triton_kernel_wrapper_functional_proxy_57 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 368, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_297, 'P_ptr': full_default_11, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_297 = None
        getitem_128: "i32[401408, 8]" = triton_kernel_wrapper_functional_proxy_57['P_ptr'];  triton_kernel_wrapper_functional_proxy_57 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_11: "bf16[256, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16);  primals_62 = None
        empty_89: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_90: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_91: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_302: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_126, [512, -1, 256])
        view_303: "bf16[401408, 256]" = torch.ops.aten.view.default(view_302, [401408, 256]);  view_302 = None
        triton_kernel_wrapper_functional_proxy_58 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 258, constant_args_idx = 369, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_303, 'P_ptr': empty_89, 'S_ptr': empty_90, 'M_ptr': empty_91, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_303 = empty_89 = empty_90 = empty_91 = None
        getitem_129: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_58['P_ptr']
        getitem_130: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_58['S_ptr']
        getitem_131: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_58['M_ptr'];  triton_kernel_wrapper_functional_proxy_58 = None
        convolution_10: "bf16[512, 256, 56, 56]" = torch.ops.aten.convolution.default(getitem_126, convert_element_type_11, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_126 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_42: "i64[]" = torch.ops.aten.add.Tensor(primals_63, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_309: "bf16[512, 256, 3136]" = torch.ops.aten.view.default(convolution_10, [512, 256, 3136]);  convolution_10 = None
        triton_kernel_wrapper_functional_proxy_59 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 259, constant_args_idx = 370, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_309, 'SUM': full_default_18, 'SUMSQ': full_default_18, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_132: "f32[256]" = triton_kernel_wrapper_functional_proxy_59['SUM']
        getitem_133: "f32[256]" = triton_kernel_wrapper_functional_proxy_59['SUMSQ'];  triton_kernel_wrapper_functional_proxy_59 = None
        div_30: "f32[256]" = torch.ops.aten.div.Tensor(getitem_132, full_default_8);  getitem_132 = None
        div_31: "f32[256]" = torch.ops.aten.div.Tensor(getitem_133, full_default_8);  getitem_133 = None
        mul_71: "f32[256]" = torch.ops.aten.mul.Tensor(div_30, div_30)
        sub_20: "f32[256]" = torch.ops.aten.sub.Tensor(div_31, mul_71);  div_31 = mul_71 = None
        clamp_min_20: "f32[256]" = torch.ops.aten.clamp_min.default(sub_20, 0.0);  sub_20 = None
        add_43: "f32[256]" = torch.ops.aten.add.Tensor(clamp_min_20, 1e-05)
        rsqrt_10: "f32[256]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        mul_72: "f32[256]" = torch.ops.aten.mul.Tensor(clamp_min_20, full_default_9);  clamp_min_20 = None
        mul_73: "f32[256]" = torch.ops.aten.mul.Tensor(primals_66, 0.9)
        mul_74: "f32[256]" = torch.ops.aten.mul.Tensor(div_30, 0.1)
        add_44: "f32[256]" = torch.ops.aten.add.Tensor(mul_73, mul_74);  mul_73 = mul_74 = None
        mul_75: "f32[256]" = torch.ops.aten.mul.Tensor(primals_67, 0.9)
        mul_76: "f32[256]" = torch.ops.aten.mul.Tensor(mul_72, 0.1);  mul_72 = None
        add_45: "f32[256]" = torch.ops.aten.add.Tensor(mul_75, mul_76);  mul_75 = mul_76 = None
        empty_92: "bf16[512, 256, 3136]" = torch.ops.aten.empty.memory_format([512, 256, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_29: "bf16[512, 256, 3136]" = torch.ops.aten.permute.default(empty_92, [0, 1, 2]);  empty_92 = None
        empty_93: "bf16[512, 256, 3136]" = torch.ops.aten.empty.memory_format([512, 256, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_30: "bf16[512, 256, 3136]" = torch.ops.aten.permute.default(empty_93, [0, 1, 2]);  empty_93 = None
        triton_kernel_wrapper_functional_proxy_60 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 260, constant_args_idx = 371, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_309, 'MEAN': div_30, 'INVSTD': rsqrt_10, 'GAMMA': primals_64, 'BETA': primals_65, 'Y': permute_29, 'X_hat': permute_30, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_309 = div_30 = primals_65 = permute_29 = permute_30 = None
        getitem_134: "bf16[512, 256, 3136]" = triton_kernel_wrapper_functional_proxy_60['Y']
        getitem_135: "bf16[512, 256, 3136]" = triton_kernel_wrapper_functional_proxy_60['X_hat'];  triton_kernel_wrapper_functional_proxy_60 = None
        empty_94: "i32[1605632, 16]" = torch.ops.aten.empty.memory_format([1605632, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_95: "bf16[1605632]" = torch.ops.aten.empty.memory_format([1605632], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_96: "bf16[1605632]" = torch.ops.aten.empty.memory_format([1605632], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_315: "bf16[512, 3136, 256]" = torch.ops.aten.view.default(getitem_135, [512, -1, 256]);  getitem_135 = None
        view_316: "bf16[1605632, 256]" = torch.ops.aten.view.default(view_315, [1605632, 256]);  view_315 = None
        triton_kernel_wrapper_functional_proxy_61 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 261, constant_args_idx = 372, grid = [(1605632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_316, 'P_ptr': empty_94, 'S_ptr': empty_95, 'M_ptr': empty_96, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_316 = empty_94 = empty_95 = empty_96 = None
        getitem_136: "i32[1605632, 16]" = triton_kernel_wrapper_functional_proxy_61['P_ptr']
        getitem_137: "bf16[1605632]" = triton_kernel_wrapper_functional_proxy_61['S_ptr']
        getitem_138: "bf16[1605632]" = triton_kernel_wrapper_functional_proxy_61['M_ptr'];  triton_kernel_wrapper_functional_proxy_61 = None
        
         # File: <eval_with_key>.3 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:39 in forward, code: add_2 = layer1_2_bn3 + layer1_1_relu_2;  layer1_2_bn3 = layer1_1_relu_2 = None
        view_322: "bf16[512, 256, 56, 56]" = torch.ops.aten.view.default(getitem_134, [512, 256, 56, 56]);  getitem_134 = None
        add_46: "bf16[512, 256, 56, 56]" = torch.ops.aten.add.Tensor(view_322, getitem_100);  view_322 = getitem_100 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        empty_97: "bf16[512, 256, 56, 56]" = torch.ops.aten.empty.memory_format([512, 256, 56, 56], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_31: "bf16[512, 256, 56, 56]" = torch.ops.aten.permute.default(empty_97, [0, 1, 2, 3]);  empty_97 = None
        triton_kernel_wrapper_functional_proxy_62 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 373, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_46, 'Y_ptr': permute_31, 'Mask_prt': full_default_26, 'n_elts': 411041792, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  add_46 = permute_31 = full_default_26 = None
        getitem_139: "bf16[512, 256, 56, 56]" = triton_kernel_wrapper_functional_proxy_62['Y_ptr']
        getitem_140: "i8[512, 256, 56, 56]" = triton_kernel_wrapper_functional_proxy_62['Mask_prt'];  triton_kernel_wrapper_functional_proxy_62 = None
        view_327: "i8[512, 3136, 256]" = torch.ops.aten.view.default(getitem_140, [512, -1, 256]);  getitem_140 = None
        view_328: "i8[1605632, 256]" = torch.ops.aten.view.default(view_327, [1605632, 256]);  view_327 = None
        triton_kernel_wrapper_functional_proxy_63 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 374, grid = [(1605632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_328, 'P_ptr': full_default_5, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_328 = full_default_5 = None
        getitem_141: "i32[1605632, 8]" = triton_kernel_wrapper_functional_proxy_63['P_ptr'];  triton_kernel_wrapper_functional_proxy_63 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_12: "bf16[128, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_68, torch.bfloat16);  primals_68 = None
        empty_98: "i32[1605632, 16]" = torch.ops.aten.empty.memory_format([1605632, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_99: "bf16[1605632]" = torch.ops.aten.empty.memory_format([1605632], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_100: "bf16[1605632]" = torch.ops.aten.empty.memory_format([1605632], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_333: "bf16[512, 3136, 256]" = torch.ops.aten.view.default(getitem_139, [512, -1, 256])
        view_334: "bf16[1605632, 256]" = torch.ops.aten.view.default(view_333, [1605632, 256]);  view_333 = None
        triton_kernel_wrapper_functional_proxy_64 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 262, constant_args_idx = 375, grid = [(1605632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_334, 'P_ptr': empty_98, 'S_ptr': empty_99, 'M_ptr': empty_100, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  empty_98 = empty_99 = empty_100 = None
        getitem_142: "i32[1605632, 16]" = triton_kernel_wrapper_functional_proxy_64['P_ptr']
        getitem_143: "bf16[1605632]" = triton_kernel_wrapper_functional_proxy_64['S_ptr']
        getitem_144: "bf16[1605632]" = triton_kernel_wrapper_functional_proxy_64['M_ptr'];  triton_kernel_wrapper_functional_proxy_64 = None
        convolution_11: "bf16[512, 128, 56, 56]" = torch.ops.aten.convolution.default(getitem_139, convert_element_type_12, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_47: "i64[]" = torch.ops.aten.add.Tensor(primals_69, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_340: "bf16[512, 128, 3136]" = torch.ops.aten.view.default(convolution_11, [512, 128, 3136]);  convolution_11 = None
        full_default_64: "f32[128]" = torch.ops.aten.full.default([128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_65 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 263, constant_args_idx = 376, grid = [(128, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_340, 'SUM': full_default_64, 'SUMSQ': full_default_64, 'M': 1605632, 'HW': 3136, 'stride_n': 401408, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_145: "f32[128]" = triton_kernel_wrapper_functional_proxy_65['SUM']
        getitem_146: "f32[128]" = triton_kernel_wrapper_functional_proxy_65['SUMSQ'];  triton_kernel_wrapper_functional_proxy_65 = None
        div_33: "f32[128]" = torch.ops.aten.div.Tensor(getitem_145, full_default_8);  getitem_145 = None
        div_34: "f32[128]" = torch.ops.aten.div.Tensor(getitem_146, full_default_8);  getitem_146 = full_default_8 = None
        mul_78: "f32[128]" = torch.ops.aten.mul.Tensor(div_33, div_33)
        sub_22: "f32[128]" = torch.ops.aten.sub.Tensor(div_34, mul_78);  div_34 = mul_78 = None
        clamp_min_22: "f32[128]" = torch.ops.aten.clamp_min.default(sub_22, 0.0);  sub_22 = None
        add_48: "f32[128]" = torch.ops.aten.add.Tensor(clamp_min_22, 1e-05)
        rsqrt_11: "f32[128]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        mul_79: "f32[128]" = torch.ops.aten.mul.Tensor(clamp_min_22, full_default_9);  clamp_min_22 = full_default_9 = None
        mul_80: "f32[128]" = torch.ops.aten.mul.Tensor(primals_72, 0.9)
        mul_81: "f32[128]" = torch.ops.aten.mul.Tensor(div_33, 0.1)
        add_49: "f32[128]" = torch.ops.aten.add.Tensor(mul_80, mul_81);  mul_80 = mul_81 = None
        mul_82: "f32[128]" = torch.ops.aten.mul.Tensor(primals_73, 0.9)
        mul_83: "f32[128]" = torch.ops.aten.mul.Tensor(mul_79, 0.1);  mul_79 = None
        add_50: "f32[128]" = torch.ops.aten.add.Tensor(mul_82, mul_83);  mul_82 = mul_83 = None
        empty_101: "bf16[512, 128, 3136]" = torch.ops.aten.empty.memory_format([512, 128, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_32: "bf16[512, 128, 3136]" = torch.ops.aten.permute.default(empty_101, [0, 1, 2]);  empty_101 = None
        empty_102: "bf16[512, 128, 3136]" = torch.ops.aten.empty.memory_format([512, 128, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_33: "bf16[512, 128, 3136]" = torch.ops.aten.permute.default(empty_102, [0, 1, 2]);  empty_102 = None
        triton_kernel_wrapper_functional_proxy_66 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 264, constant_args_idx = 377, grid = [(128, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_340, 'MEAN': div_33, 'INVSTD': rsqrt_11, 'GAMMA': primals_70, 'BETA': primals_71, 'Y': permute_32, 'X_hat': permute_33, 'M': 1605632, 'HW': 3136, 'stride_n': 401408, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_340 = div_33 = primals_71 = permute_32 = permute_33 = None
        getitem_147: "bf16[512, 128, 3136]" = triton_kernel_wrapper_functional_proxy_66['Y']
        getitem_148: "bf16[512, 128, 3136]" = triton_kernel_wrapper_functional_proxy_66['X_hat'];  triton_kernel_wrapper_functional_proxy_66 = None
        empty_103: "i32[802816, 16]" = torch.ops.aten.empty.memory_format([802816, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_104: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_105: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_346: "bf16[512, 1568, 256]" = torch.ops.aten.view.default(getitem_148, [512, -1, 256]);  getitem_148 = None
        view_347: "bf16[802816, 256]" = torch.ops.aten.view.default(view_346, [802816, 256]);  view_346 = None
        triton_kernel_wrapper_functional_proxy_67 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 265, constant_args_idx = 378, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_347, 'P_ptr': empty_103, 'S_ptr': empty_104, 'M_ptr': empty_105, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_347 = empty_103 = empty_104 = empty_105 = None
        getitem_149: "i32[802816, 16]" = triton_kernel_wrapper_functional_proxy_67['P_ptr']
        getitem_150: "bf16[802816]" = triton_kernel_wrapper_functional_proxy_67['S_ptr']
        getitem_151: "bf16[802816]" = triton_kernel_wrapper_functional_proxy_67['M_ptr'];  triton_kernel_wrapper_functional_proxy_67 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_68: "i8[512, 128, 56, 56]" = torch.ops.aten.full.default([512, 128, 56, 56], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_353: "bf16[512, 128, 56, 56]" = torch.ops.aten.view.default(getitem_147, [512, 128, 56, 56]);  getitem_147 = None
        empty_106: "bf16[512, 128, 56, 56]" = torch.ops.aten.empty.memory_format([512, 128, 56, 56], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_34: "bf16[512, 128, 56, 56]" = torch.ops.aten.permute.default(empty_106, [0, 1, 2, 3]);  empty_106 = None
        triton_kernel_wrapper_functional_proxy_68 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 379, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_353, 'Y_ptr': permute_34, 'Mask_prt': full_default_68, 'n_elts': 205520896, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_353 = permute_34 = full_default_68 = None
        getitem_152: "bf16[512, 128, 56, 56]" = triton_kernel_wrapper_functional_proxy_68['Y_ptr']
        getitem_153: "i8[512, 128, 56, 56]" = triton_kernel_wrapper_functional_proxy_68['Mask_prt'];  triton_kernel_wrapper_functional_proxy_68 = None
        full_default_69: "i32[802816, 8]" = torch.ops.aten.full.default([802816, 8], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_358: "i8[512, 1568, 256]" = torch.ops.aten.view.default(getitem_153, [512, -1, 256]);  getitem_153 = None
        view_359: "i8[802816, 256]" = torch.ops.aten.view.default(view_358, [802816, 256]);  view_358 = None
        triton_kernel_wrapper_functional_proxy_69 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 380, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_359, 'P_ptr': full_default_69, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_359 = None
        getitem_154: "i32[802816, 8]" = triton_kernel_wrapper_functional_proxy_69['P_ptr'];  triton_kernel_wrapper_functional_proxy_69 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_13: "bf16[128, 128, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_74, torch.bfloat16);  primals_74 = None
        empty_107: "i32[802816, 16]" = torch.ops.aten.empty.memory_format([802816, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_108: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_109: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_364: "bf16[512, 1568, 256]" = torch.ops.aten.view.default(getitem_152, [512, -1, 256])
        view_365: "bf16[802816, 256]" = torch.ops.aten.view.default(view_364, [802816, 256]);  view_364 = None
        triton_kernel_wrapper_functional_proxy_70 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 266, constant_args_idx = 381, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_365, 'P_ptr': empty_107, 'S_ptr': empty_108, 'M_ptr': empty_109, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_365 = empty_107 = empty_108 = empty_109 = None
        getitem_155: "i32[802816, 16]" = triton_kernel_wrapper_functional_proxy_70['P_ptr']
        getitem_156: "bf16[802816]" = triton_kernel_wrapper_functional_proxy_70['S_ptr']
        getitem_157: "bf16[802816]" = triton_kernel_wrapper_functional_proxy_70['M_ptr'];  triton_kernel_wrapper_functional_proxy_70 = None
        convolution_12: "bf16[512, 128, 28, 28]" = torch.ops.aten.convolution.default(getitem_152, convert_element_type_13, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_152 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_51: "i64[]" = torch.ops.aten.add.Tensor(primals_75, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_371: "bf16[512, 128, 784]" = torch.ops.aten.view.default(convolution_12, [512, 128, 784]);  convolution_12 = None
        triton_kernel_wrapper_functional_proxy_71 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 267, constant_args_idx = 382, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_371, 'SUM': full_default_64, 'SUMSQ': full_default_64, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_158: "f32[128]" = triton_kernel_wrapper_functional_proxy_71['SUM']
        getitem_159: "f32[128]" = triton_kernel_wrapper_functional_proxy_71['SUMSQ'];  triton_kernel_wrapper_functional_proxy_71 = None
        full_default_72: "f32[]" = torch.ops.aten.full.default([], 401408.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        div_36: "f32[128]" = torch.ops.aten.div.Tensor(getitem_158, full_default_72);  getitem_158 = None
        div_37: "f32[128]" = torch.ops.aten.div.Tensor(getitem_159, full_default_72);  getitem_159 = None
        mul_85: "f32[128]" = torch.ops.aten.mul.Tensor(div_36, div_36)
        sub_24: "f32[128]" = torch.ops.aten.sub.Tensor(div_37, mul_85);  div_37 = mul_85 = None
        clamp_min_24: "f32[128]" = torch.ops.aten.clamp_min.default(sub_24, 0.0);  sub_24 = None
        add_52: "f32[128]" = torch.ops.aten.add.Tensor(clamp_min_24, 1e-05)
        rsqrt_12: "f32[128]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        full_default_73: "f32[]" = torch.ops.aten.full.default([], 1.0000025033950806, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        mul_86: "f32[128]" = torch.ops.aten.mul.Tensor(clamp_min_24, full_default_73);  clamp_min_24 = None
        mul_87: "f32[128]" = torch.ops.aten.mul.Tensor(primals_78, 0.9)
        mul_88: "f32[128]" = torch.ops.aten.mul.Tensor(div_36, 0.1)
        add_53: "f32[128]" = torch.ops.aten.add.Tensor(mul_87, mul_88);  mul_87 = mul_88 = None
        mul_89: "f32[128]" = torch.ops.aten.mul.Tensor(primals_79, 0.9)
        mul_90: "f32[128]" = torch.ops.aten.mul.Tensor(mul_86, 0.1);  mul_86 = None
        add_54: "f32[128]" = torch.ops.aten.add.Tensor(mul_89, mul_90);  mul_89 = mul_90 = None
        empty_110: "bf16[512, 128, 784]" = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_35: "bf16[512, 128, 784]" = torch.ops.aten.permute.default(empty_110, [0, 1, 2]);  empty_110 = None
        empty_111: "bf16[512, 128, 784]" = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_36: "bf16[512, 128, 784]" = torch.ops.aten.permute.default(empty_111, [0, 1, 2]);  empty_111 = None
        triton_kernel_wrapper_functional_proxy_72 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 268, constant_args_idx = 383, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_371, 'MEAN': div_36, 'INVSTD': rsqrt_12, 'GAMMA': primals_76, 'BETA': primals_77, 'Y': permute_35, 'X_hat': permute_36, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_371 = div_36 = primals_77 = permute_35 = permute_36 = None
        getitem_160: "bf16[512, 128, 784]" = triton_kernel_wrapper_functional_proxy_72['Y']
        getitem_161: "bf16[512, 128, 784]" = triton_kernel_wrapper_functional_proxy_72['X_hat'];  triton_kernel_wrapper_functional_proxy_72 = None
        empty_112: "i32[200704, 16]" = torch.ops.aten.empty.memory_format([200704, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_113: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_114: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_377: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_161, [512, -1, 256]);  getitem_161 = None
        view_378: "bf16[200704, 256]" = torch.ops.aten.view.default(view_377, [200704, 256]);  view_377 = None
        triton_kernel_wrapper_functional_proxy_73 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 269, constant_args_idx = 384, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_378, 'P_ptr': empty_112, 'S_ptr': empty_113, 'M_ptr': empty_114, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_378 = empty_112 = empty_113 = empty_114 = None
        getitem_162: "i32[200704, 16]" = triton_kernel_wrapper_functional_proxy_73['P_ptr']
        getitem_163: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_73['S_ptr']
        getitem_164: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_73['M_ptr'];  triton_kernel_wrapper_functional_proxy_73 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_74: "i8[512, 128, 28, 28]" = torch.ops.aten.full.default([512, 128, 28, 28], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_384: "bf16[512, 128, 28, 28]" = torch.ops.aten.view.default(getitem_160, [512, 128, 28, 28]);  getitem_160 = None
        empty_115: "bf16[512, 128, 28, 28]" = torch.ops.aten.empty.memory_format([512, 128, 28, 28], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_37: "bf16[512, 128, 28, 28]" = torch.ops.aten.permute.default(empty_115, [0, 1, 2, 3]);  empty_115 = None
        triton_kernel_wrapper_functional_proxy_74 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 385, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_384, 'Y_ptr': permute_37, 'Mask_prt': full_default_74, 'n_elts': 51380224, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_384 = permute_37 = None
        getitem_165: "bf16[512, 128, 28, 28]" = triton_kernel_wrapper_functional_proxy_74['Y_ptr']
        getitem_166: "i8[512, 128, 28, 28]" = triton_kernel_wrapper_functional_proxy_74['Mask_prt'];  triton_kernel_wrapper_functional_proxy_74 = None
        full_default_75: "i32[200704, 8]" = torch.ops.aten.full.default([200704, 8], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_389: "i8[512, 392, 256]" = torch.ops.aten.view.default(getitem_166, [512, -1, 256]);  getitem_166 = None
        view_390: "i8[200704, 256]" = torch.ops.aten.view.default(view_389, [200704, 256]);  view_389 = None
        triton_kernel_wrapper_functional_proxy_75 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 386, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_390, 'P_ptr': full_default_75, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_390 = None
        getitem_167: "i32[200704, 8]" = triton_kernel_wrapper_functional_proxy_75['P_ptr'];  triton_kernel_wrapper_functional_proxy_75 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_14: "bf16[512, 128, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_80, torch.bfloat16);  primals_80 = None
        empty_116: "i32[200704, 16]" = torch.ops.aten.empty.memory_format([200704, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_117: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_118: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_395: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_165, [512, -1, 256])
        view_396: "bf16[200704, 256]" = torch.ops.aten.view.default(view_395, [200704, 256]);  view_395 = None
        triton_kernel_wrapper_functional_proxy_76 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 270, constant_args_idx = 387, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_396, 'P_ptr': empty_116, 'S_ptr': empty_117, 'M_ptr': empty_118, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_396 = empty_116 = empty_117 = empty_118 = None
        getitem_168: "i32[200704, 16]" = triton_kernel_wrapper_functional_proxy_76['P_ptr']
        getitem_169: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_76['S_ptr']
        getitem_170: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_76['M_ptr'];  triton_kernel_wrapper_functional_proxy_76 = None
        convolution_13: "bf16[512, 512, 28, 28]" = torch.ops.aten.convolution.default(getitem_165, convert_element_type_14, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_165 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_55: "i64[]" = torch.ops.aten.add.Tensor(primals_81, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_402: "bf16[512, 512, 784]" = torch.ops.aten.view.default(convolution_13, [512, 512, 784]);  convolution_13 = None
        full_default_76: "f32[512]" = torch.ops.aten.full.default([512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_77 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 271, constant_args_idx = 388, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_402, 'SUM': full_default_76, 'SUMSQ': full_default_76, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_171: "f32[512]" = triton_kernel_wrapper_functional_proxy_77['SUM']
        getitem_172: "f32[512]" = triton_kernel_wrapper_functional_proxy_77['SUMSQ'];  triton_kernel_wrapper_functional_proxy_77 = None
        div_39: "f32[512]" = torch.ops.aten.div.Tensor(getitem_171, full_default_72);  getitem_171 = None
        div_40: "f32[512]" = torch.ops.aten.div.Tensor(getitem_172, full_default_72);  getitem_172 = None
        mul_92: "f32[512]" = torch.ops.aten.mul.Tensor(div_39, div_39)
        sub_26: "f32[512]" = torch.ops.aten.sub.Tensor(div_40, mul_92);  div_40 = mul_92 = None
        clamp_min_26: "f32[512]" = torch.ops.aten.clamp_min.default(sub_26, 0.0);  sub_26 = None
        add_56: "f32[512]" = torch.ops.aten.add.Tensor(clamp_min_26, 1e-05)
        rsqrt_13: "f32[512]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_93: "f32[512]" = torch.ops.aten.mul.Tensor(clamp_min_26, full_default_73);  clamp_min_26 = None
        mul_94: "f32[512]" = torch.ops.aten.mul.Tensor(primals_84, 0.9)
        mul_95: "f32[512]" = torch.ops.aten.mul.Tensor(div_39, 0.1)
        add_57: "f32[512]" = torch.ops.aten.add.Tensor(mul_94, mul_95);  mul_94 = mul_95 = None
        mul_96: "f32[512]" = torch.ops.aten.mul.Tensor(primals_85, 0.9)
        mul_97: "f32[512]" = torch.ops.aten.mul.Tensor(mul_93, 0.1);  mul_93 = None
        add_58: "f32[512]" = torch.ops.aten.add.Tensor(mul_96, mul_97);  mul_96 = mul_97 = None
        empty_119: "bf16[512, 512, 784]" = torch.ops.aten.empty.memory_format([512, 512, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_38: "bf16[512, 512, 784]" = torch.ops.aten.permute.default(empty_119, [0, 1, 2]);  empty_119 = None
        empty_120: "bf16[512, 512, 784]" = torch.ops.aten.empty.memory_format([512, 512, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_39: "bf16[512, 512, 784]" = torch.ops.aten.permute.default(empty_120, [0, 1, 2]);  empty_120 = None
        triton_kernel_wrapper_functional_proxy_78 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 272, constant_args_idx = 389, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_402, 'MEAN': div_39, 'INVSTD': rsqrt_13, 'GAMMA': primals_82, 'BETA': primals_83, 'Y': permute_38, 'X_hat': permute_39, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_402 = div_39 = primals_83 = permute_38 = permute_39 = None
        getitem_173: "bf16[512, 512, 784]" = triton_kernel_wrapper_functional_proxy_78['Y']
        getitem_174: "bf16[512, 512, 784]" = triton_kernel_wrapper_functional_proxy_78['X_hat'];  triton_kernel_wrapper_functional_proxy_78 = None
        empty_121: "i32[802816, 16]" = torch.ops.aten.empty.memory_format([802816, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_122: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_123: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_408: "bf16[512, 1568, 256]" = torch.ops.aten.view.default(getitem_174, [512, -1, 256]);  getitem_174 = None
        view_409: "bf16[802816, 256]" = torch.ops.aten.view.default(view_408, [802816, 256]);  view_408 = None
        triton_kernel_wrapper_functional_proxy_79 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 273, constant_args_idx = 390, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_409, 'P_ptr': empty_121, 'S_ptr': empty_122, 'M_ptr': empty_123, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_409 = empty_121 = empty_122 = empty_123 = None
        getitem_175: "i32[802816, 16]" = triton_kernel_wrapper_functional_proxy_79['P_ptr']
        getitem_176: "bf16[802816]" = triton_kernel_wrapper_functional_proxy_79['S_ptr']
        getitem_177: "bf16[802816]" = triton_kernel_wrapper_functional_proxy_79['M_ptr'];  triton_kernel_wrapper_functional_proxy_79 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_15: "bf16[512, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_86, torch.bfloat16);  primals_86 = None
        empty_124: "i32[1605632, 16]" = torch.ops.aten.empty.memory_format([1605632, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_125: "bf16[1605632]" = torch.ops.aten.empty.memory_format([1605632], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_126: "bf16[1605632]" = torch.ops.aten.empty.memory_format([1605632], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_80 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 274, constant_args_idx = 391, grid = [(1605632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_334, 'P_ptr': empty_124, 'S_ptr': empty_125, 'M_ptr': empty_126, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_334 = empty_124 = empty_125 = empty_126 = None
        getitem_178: "i32[1605632, 16]" = triton_kernel_wrapper_functional_proxy_80['P_ptr']
        getitem_179: "bf16[1605632]" = triton_kernel_wrapper_functional_proxy_80['S_ptr']
        getitem_180: "bf16[1605632]" = triton_kernel_wrapper_functional_proxy_80['M_ptr'];  triton_kernel_wrapper_functional_proxy_80 = None
        convolution_14: "bf16[512, 512, 28, 28]" = torch.ops.aten.convolution.default(getitem_139, convert_element_type_15, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  getitem_139 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_59: "i64[]" = torch.ops.aten.add.Tensor(primals_87, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_426: "bf16[512, 512, 784]" = torch.ops.aten.view.default(convolution_14, [512, 512, 784]);  convolution_14 = None
        triton_kernel_wrapper_functional_proxy_81 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 275, constant_args_idx = 392, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_426, 'SUM': full_default_76, 'SUMSQ': full_default_76, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_181: "f32[512]" = triton_kernel_wrapper_functional_proxy_81['SUM']
        getitem_182: "f32[512]" = triton_kernel_wrapper_functional_proxy_81['SUMSQ'];  triton_kernel_wrapper_functional_proxy_81 = None
        div_42: "f32[512]" = torch.ops.aten.div.Tensor(getitem_181, full_default_72);  getitem_181 = None
        div_43: "f32[512]" = torch.ops.aten.div.Tensor(getitem_182, full_default_72);  getitem_182 = None
        mul_99: "f32[512]" = torch.ops.aten.mul.Tensor(div_42, div_42)
        sub_28: "f32[512]" = torch.ops.aten.sub.Tensor(div_43, mul_99);  div_43 = mul_99 = None
        clamp_min_28: "f32[512]" = torch.ops.aten.clamp_min.default(sub_28, 0.0);  sub_28 = None
        add_60: "f32[512]" = torch.ops.aten.add.Tensor(clamp_min_28, 1e-05)
        rsqrt_14: "f32[512]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_100: "f32[512]" = torch.ops.aten.mul.Tensor(clamp_min_28, full_default_73);  clamp_min_28 = None
        mul_101: "f32[512]" = torch.ops.aten.mul.Tensor(primals_90, 0.9)
        mul_102: "f32[512]" = torch.ops.aten.mul.Tensor(div_42, 0.1)
        add_61: "f32[512]" = torch.ops.aten.add.Tensor(mul_101, mul_102);  mul_101 = mul_102 = None
        mul_103: "f32[512]" = torch.ops.aten.mul.Tensor(primals_91, 0.9)
        mul_104: "f32[512]" = torch.ops.aten.mul.Tensor(mul_100, 0.1);  mul_100 = None
        add_62: "f32[512]" = torch.ops.aten.add.Tensor(mul_103, mul_104);  mul_103 = mul_104 = None
        empty_127: "bf16[512, 512, 784]" = torch.ops.aten.empty.memory_format([512, 512, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_40: "bf16[512, 512, 784]" = torch.ops.aten.permute.default(empty_127, [0, 1, 2]);  empty_127 = None
        empty_128: "bf16[512, 512, 784]" = torch.ops.aten.empty.memory_format([512, 512, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_41: "bf16[512, 512, 784]" = torch.ops.aten.permute.default(empty_128, [0, 1, 2]);  empty_128 = None
        triton_kernel_wrapper_functional_proxy_82 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 276, constant_args_idx = 393, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_426, 'MEAN': div_42, 'INVSTD': rsqrt_14, 'GAMMA': primals_88, 'BETA': primals_89, 'Y': permute_40, 'X_hat': permute_41, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_426 = div_42 = primals_89 = permute_40 = permute_41 = None
        getitem_183: "bf16[512, 512, 784]" = triton_kernel_wrapper_functional_proxy_82['Y']
        getitem_184: "bf16[512, 512, 784]" = triton_kernel_wrapper_functional_proxy_82['X_hat'];  triton_kernel_wrapper_functional_proxy_82 = None
        empty_129: "i32[802816, 16]" = torch.ops.aten.empty.memory_format([802816, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_130: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_131: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_432: "bf16[512, 1568, 256]" = torch.ops.aten.view.default(getitem_184, [512, -1, 256]);  getitem_184 = None
        view_433: "bf16[802816, 256]" = torch.ops.aten.view.default(view_432, [802816, 256]);  view_432 = None
        triton_kernel_wrapper_functional_proxy_83 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 277, constant_args_idx = 394, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_433, 'P_ptr': empty_129, 'S_ptr': empty_130, 'M_ptr': empty_131, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_433 = empty_129 = empty_130 = empty_131 = None
        getitem_185: "i32[802816, 16]" = triton_kernel_wrapper_functional_proxy_83['P_ptr']
        getitem_186: "bf16[802816]" = triton_kernel_wrapper_functional_proxy_83['S_ptr']
        getitem_187: "bf16[802816]" = triton_kernel_wrapper_functional_proxy_83['M_ptr'];  triton_kernel_wrapper_functional_proxy_83 = None
        
         # File: <eval_with_key>.3 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:51 in forward, code: add_3 = layer2_0_bn3 + layer2_0_downsample_1;  layer2_0_bn3 = layer2_0_downsample_1 = None
        view_439: "bf16[512, 512, 28, 28]" = torch.ops.aten.view.default(getitem_173, [512, 512, 28, 28]);  getitem_173 = None
        view_440: "bf16[512, 512, 28, 28]" = torch.ops.aten.view.default(getitem_183, [512, 512, 28, 28]);  getitem_183 = None
        add_63: "bf16[512, 512, 28, 28]" = torch.ops.aten.add.Tensor(view_439, view_440);  view_439 = view_440 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_84: "i8[512, 512, 28, 28]" = torch.ops.aten.full.default([512, 512, 28, 28], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        empty_132: "bf16[512, 512, 28, 28]" = torch.ops.aten.empty.memory_format([512, 512, 28, 28], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_42: "bf16[512, 512, 28, 28]" = torch.ops.aten.permute.default(empty_132, [0, 1, 2, 3]);  empty_132 = None
        triton_kernel_wrapper_functional_proxy_84 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 395, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_63, 'Y_ptr': permute_42, 'Mask_prt': full_default_84, 'n_elts': 205520896, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  add_63 = permute_42 = None
        getitem_188: "bf16[512, 512, 28, 28]" = triton_kernel_wrapper_functional_proxy_84['Y_ptr']
        getitem_189: "i8[512, 512, 28, 28]" = triton_kernel_wrapper_functional_proxy_84['Mask_prt'];  triton_kernel_wrapper_functional_proxy_84 = None
        view_445: "i8[512, 1568, 256]" = torch.ops.aten.view.default(getitem_189, [512, -1, 256]);  getitem_189 = None
        view_446: "i8[802816, 256]" = torch.ops.aten.view.default(view_445, [802816, 256]);  view_445 = None
        triton_kernel_wrapper_functional_proxy_85 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 396, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_446, 'P_ptr': full_default_69, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_446 = None
        getitem_190: "i32[802816, 8]" = triton_kernel_wrapper_functional_proxy_85['P_ptr'];  triton_kernel_wrapper_functional_proxy_85 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_16: "bf16[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_92, torch.bfloat16);  primals_92 = None
        empty_133: "i32[802816, 16]" = torch.ops.aten.empty.memory_format([802816, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_134: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_135: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_451: "bf16[512, 1568, 256]" = torch.ops.aten.view.default(getitem_188, [512, -1, 256])
        view_452: "bf16[802816, 256]" = torch.ops.aten.view.default(view_451, [802816, 256]);  view_451 = None
        triton_kernel_wrapper_functional_proxy_86 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 278, constant_args_idx = 397, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_452, 'P_ptr': empty_133, 'S_ptr': empty_134, 'M_ptr': empty_135, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_452 = empty_133 = empty_134 = empty_135 = None
        getitem_191: "i32[802816, 16]" = triton_kernel_wrapper_functional_proxy_86['P_ptr']
        getitem_192: "bf16[802816]" = triton_kernel_wrapper_functional_proxy_86['S_ptr']
        getitem_193: "bf16[802816]" = triton_kernel_wrapper_functional_proxy_86['M_ptr'];  triton_kernel_wrapper_functional_proxy_86 = None
        convolution_15: "bf16[512, 128, 28, 28]" = torch.ops.aten.convolution.default(getitem_188, convert_element_type_16, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_64: "i64[]" = torch.ops.aten.add.Tensor(primals_93, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_458: "bf16[512, 128, 784]" = torch.ops.aten.view.default(convolution_15, [512, 128, 784]);  convolution_15 = None
        triton_kernel_wrapper_functional_proxy_87 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 279, constant_args_idx = 398, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_458, 'SUM': full_default_64, 'SUMSQ': full_default_64, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_194: "f32[128]" = triton_kernel_wrapper_functional_proxy_87['SUM']
        getitem_195: "f32[128]" = triton_kernel_wrapper_functional_proxy_87['SUMSQ'];  triton_kernel_wrapper_functional_proxy_87 = None
        div_45: "f32[128]" = torch.ops.aten.div.Tensor(getitem_194, full_default_72);  getitem_194 = None
        div_46: "f32[128]" = torch.ops.aten.div.Tensor(getitem_195, full_default_72);  getitem_195 = None
        mul_106: "f32[128]" = torch.ops.aten.mul.Tensor(div_45, div_45)
        sub_30: "f32[128]" = torch.ops.aten.sub.Tensor(div_46, mul_106);  div_46 = mul_106 = None
        clamp_min_30: "f32[128]" = torch.ops.aten.clamp_min.default(sub_30, 0.0);  sub_30 = None
        add_65: "f32[128]" = torch.ops.aten.add.Tensor(clamp_min_30, 1e-05)
        rsqrt_15: "f32[128]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        mul_107: "f32[128]" = torch.ops.aten.mul.Tensor(clamp_min_30, full_default_73);  clamp_min_30 = None
        mul_108: "f32[128]" = torch.ops.aten.mul.Tensor(primals_96, 0.9)
        mul_109: "f32[128]" = torch.ops.aten.mul.Tensor(div_45, 0.1)
        add_66: "f32[128]" = torch.ops.aten.add.Tensor(mul_108, mul_109);  mul_108 = mul_109 = None
        mul_110: "f32[128]" = torch.ops.aten.mul.Tensor(primals_97, 0.9)
        mul_111: "f32[128]" = torch.ops.aten.mul.Tensor(mul_107, 0.1);  mul_107 = None
        add_67: "f32[128]" = torch.ops.aten.add.Tensor(mul_110, mul_111);  mul_110 = mul_111 = None
        empty_136: "bf16[512, 128, 784]" = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_43: "bf16[512, 128, 784]" = torch.ops.aten.permute.default(empty_136, [0, 1, 2]);  empty_136 = None
        empty_137: "bf16[512, 128, 784]" = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_44: "bf16[512, 128, 784]" = torch.ops.aten.permute.default(empty_137, [0, 1, 2]);  empty_137 = None
        triton_kernel_wrapper_functional_proxy_88 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 280, constant_args_idx = 399, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_458, 'MEAN': div_45, 'INVSTD': rsqrt_15, 'GAMMA': primals_94, 'BETA': primals_95, 'Y': permute_43, 'X_hat': permute_44, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_458 = div_45 = primals_95 = permute_43 = permute_44 = None
        getitem_196: "bf16[512, 128, 784]" = triton_kernel_wrapper_functional_proxy_88['Y']
        getitem_197: "bf16[512, 128, 784]" = triton_kernel_wrapper_functional_proxy_88['X_hat'];  triton_kernel_wrapper_functional_proxy_88 = None
        empty_138: "i32[200704, 16]" = torch.ops.aten.empty.memory_format([200704, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_139: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_140: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_464: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_197, [512, -1, 256]);  getitem_197 = None
        view_465: "bf16[200704, 256]" = torch.ops.aten.view.default(view_464, [200704, 256]);  view_464 = None
        triton_kernel_wrapper_functional_proxy_89 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 281, constant_args_idx = 400, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_465, 'P_ptr': empty_138, 'S_ptr': empty_139, 'M_ptr': empty_140, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_465 = empty_138 = empty_139 = empty_140 = None
        getitem_198: "i32[200704, 16]" = triton_kernel_wrapper_functional_proxy_89['P_ptr']
        getitem_199: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_89['S_ptr']
        getitem_200: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_89['M_ptr'];  triton_kernel_wrapper_functional_proxy_89 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_471: "bf16[512, 128, 28, 28]" = torch.ops.aten.view.default(getitem_196, [512, 128, 28, 28]);  getitem_196 = None
        empty_141: "bf16[512, 128, 28, 28]" = torch.ops.aten.empty.memory_format([512, 128, 28, 28], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_45: "bf16[512, 128, 28, 28]" = torch.ops.aten.permute.default(empty_141, [0, 1, 2, 3]);  empty_141 = None
        triton_kernel_wrapper_functional_proxy_90 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 401, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_471, 'Y_ptr': permute_45, 'Mask_prt': full_default_74, 'n_elts': 51380224, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_471 = permute_45 = None
        getitem_201: "bf16[512, 128, 28, 28]" = triton_kernel_wrapper_functional_proxy_90['Y_ptr']
        getitem_202: "i8[512, 128, 28, 28]" = triton_kernel_wrapper_functional_proxy_90['Mask_prt'];  triton_kernel_wrapper_functional_proxy_90 = None
        view_476: "i8[512, 392, 256]" = torch.ops.aten.view.default(getitem_202, [512, -1, 256]);  getitem_202 = None
        view_477: "i8[200704, 256]" = torch.ops.aten.view.default(view_476, [200704, 256]);  view_476 = None
        triton_kernel_wrapper_functional_proxy_91 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 402, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_477, 'P_ptr': full_default_75, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_477 = None
        getitem_203: "i32[200704, 8]" = triton_kernel_wrapper_functional_proxy_91['P_ptr'];  triton_kernel_wrapper_functional_proxy_91 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_17: "bf16[128, 128, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_98, torch.bfloat16);  primals_98 = None
        empty_142: "i32[200704, 16]" = torch.ops.aten.empty.memory_format([200704, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_143: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_144: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_482: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_201, [512, -1, 256])
        view_483: "bf16[200704, 256]" = torch.ops.aten.view.default(view_482, [200704, 256]);  view_482 = None
        triton_kernel_wrapper_functional_proxy_92 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 282, constant_args_idx = 403, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_483, 'P_ptr': empty_142, 'S_ptr': empty_143, 'M_ptr': empty_144, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_483 = empty_142 = empty_143 = empty_144 = None
        getitem_204: "i32[200704, 16]" = triton_kernel_wrapper_functional_proxy_92['P_ptr']
        getitem_205: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_92['S_ptr']
        getitem_206: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_92['M_ptr'];  triton_kernel_wrapper_functional_proxy_92 = None
        convolution_16: "bf16[512, 128, 28, 28]" = torch.ops.aten.convolution.default(getitem_201, convert_element_type_17, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_201 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_68: "i64[]" = torch.ops.aten.add.Tensor(primals_99, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_489: "bf16[512, 128, 784]" = torch.ops.aten.view.default(convolution_16, [512, 128, 784]);  convolution_16 = None
        triton_kernel_wrapper_functional_proxy_93 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 283, constant_args_idx = 404, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_489, 'SUM': full_default_64, 'SUMSQ': full_default_64, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_207: "f32[128]" = triton_kernel_wrapper_functional_proxy_93['SUM']
        getitem_208: "f32[128]" = triton_kernel_wrapper_functional_proxy_93['SUMSQ'];  triton_kernel_wrapper_functional_proxy_93 = None
        div_48: "f32[128]" = torch.ops.aten.div.Tensor(getitem_207, full_default_72);  getitem_207 = None
        div_49: "f32[128]" = torch.ops.aten.div.Tensor(getitem_208, full_default_72);  getitem_208 = None
        mul_113: "f32[128]" = torch.ops.aten.mul.Tensor(div_48, div_48)
        sub_32: "f32[128]" = torch.ops.aten.sub.Tensor(div_49, mul_113);  div_49 = mul_113 = None
        clamp_min_32: "f32[128]" = torch.ops.aten.clamp_min.default(sub_32, 0.0);  sub_32 = None
        add_69: "f32[128]" = torch.ops.aten.add.Tensor(clamp_min_32, 1e-05)
        rsqrt_16: "f32[128]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        mul_114: "f32[128]" = torch.ops.aten.mul.Tensor(clamp_min_32, full_default_73);  clamp_min_32 = None
        mul_115: "f32[128]" = torch.ops.aten.mul.Tensor(primals_102, 0.9)
        mul_116: "f32[128]" = torch.ops.aten.mul.Tensor(div_48, 0.1)
        add_70: "f32[128]" = torch.ops.aten.add.Tensor(mul_115, mul_116);  mul_115 = mul_116 = None
        mul_117: "f32[128]" = torch.ops.aten.mul.Tensor(primals_103, 0.9)
        mul_118: "f32[128]" = torch.ops.aten.mul.Tensor(mul_114, 0.1);  mul_114 = None
        add_71: "f32[128]" = torch.ops.aten.add.Tensor(mul_117, mul_118);  mul_117 = mul_118 = None
        empty_145: "bf16[512, 128, 784]" = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_46: "bf16[512, 128, 784]" = torch.ops.aten.permute.default(empty_145, [0, 1, 2]);  empty_145 = None
        empty_146: "bf16[512, 128, 784]" = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_47: "bf16[512, 128, 784]" = torch.ops.aten.permute.default(empty_146, [0, 1, 2]);  empty_146 = None
        triton_kernel_wrapper_functional_proxy_94 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 284, constant_args_idx = 405, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_489, 'MEAN': div_48, 'INVSTD': rsqrt_16, 'GAMMA': primals_100, 'BETA': primals_101, 'Y': permute_46, 'X_hat': permute_47, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_489 = div_48 = primals_101 = permute_46 = permute_47 = None
        getitem_209: "bf16[512, 128, 784]" = triton_kernel_wrapper_functional_proxy_94['Y']
        getitem_210: "bf16[512, 128, 784]" = triton_kernel_wrapper_functional_proxy_94['X_hat'];  triton_kernel_wrapper_functional_proxy_94 = None
        empty_147: "i32[200704, 16]" = torch.ops.aten.empty.memory_format([200704, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_148: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_149: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_495: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_210, [512, -1, 256]);  getitem_210 = None
        view_496: "bf16[200704, 256]" = torch.ops.aten.view.default(view_495, [200704, 256]);  view_495 = None
        triton_kernel_wrapper_functional_proxy_95 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 285, constant_args_idx = 406, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_496, 'P_ptr': empty_147, 'S_ptr': empty_148, 'M_ptr': empty_149, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_496 = empty_147 = empty_148 = empty_149 = None
        getitem_211: "i32[200704, 16]" = triton_kernel_wrapper_functional_proxy_95['P_ptr']
        getitem_212: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_95['S_ptr']
        getitem_213: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_95['M_ptr'];  triton_kernel_wrapper_functional_proxy_95 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_502: "bf16[512, 128, 28, 28]" = torch.ops.aten.view.default(getitem_209, [512, 128, 28, 28]);  getitem_209 = None
        empty_150: "bf16[512, 128, 28, 28]" = torch.ops.aten.empty.memory_format([512, 128, 28, 28], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_48: "bf16[512, 128, 28, 28]" = torch.ops.aten.permute.default(empty_150, [0, 1, 2, 3]);  empty_150 = None
        triton_kernel_wrapper_functional_proxy_96 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 407, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_502, 'Y_ptr': permute_48, 'Mask_prt': full_default_74, 'n_elts': 51380224, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_502 = permute_48 = None
        getitem_214: "bf16[512, 128, 28, 28]" = triton_kernel_wrapper_functional_proxy_96['Y_ptr']
        getitem_215: "i8[512, 128, 28, 28]" = triton_kernel_wrapper_functional_proxy_96['Mask_prt'];  triton_kernel_wrapper_functional_proxy_96 = None
        view_507: "i8[512, 392, 256]" = torch.ops.aten.view.default(getitem_215, [512, -1, 256]);  getitem_215 = None
        view_508: "i8[200704, 256]" = torch.ops.aten.view.default(view_507, [200704, 256]);  view_507 = None
        triton_kernel_wrapper_functional_proxy_97 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 408, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_508, 'P_ptr': full_default_75, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_508 = None
        getitem_216: "i32[200704, 8]" = triton_kernel_wrapper_functional_proxy_97['P_ptr'];  triton_kernel_wrapper_functional_proxy_97 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_18: "bf16[512, 128, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_104, torch.bfloat16);  primals_104 = None
        empty_151: "i32[200704, 16]" = torch.ops.aten.empty.memory_format([200704, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_152: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_153: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_513: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_214, [512, -1, 256])
        view_514: "bf16[200704, 256]" = torch.ops.aten.view.default(view_513, [200704, 256]);  view_513 = None
        triton_kernel_wrapper_functional_proxy_98 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 286, constant_args_idx = 409, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_514, 'P_ptr': empty_151, 'S_ptr': empty_152, 'M_ptr': empty_153, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_514 = empty_151 = empty_152 = empty_153 = None
        getitem_217: "i32[200704, 16]" = triton_kernel_wrapper_functional_proxy_98['P_ptr']
        getitem_218: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_98['S_ptr']
        getitem_219: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_98['M_ptr'];  triton_kernel_wrapper_functional_proxy_98 = None
        convolution_17: "bf16[512, 512, 28, 28]" = torch.ops.aten.convolution.default(getitem_214, convert_element_type_18, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_214 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_72: "i64[]" = torch.ops.aten.add.Tensor(primals_105, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_520: "bf16[512, 512, 784]" = torch.ops.aten.view.default(convolution_17, [512, 512, 784]);  convolution_17 = None
        triton_kernel_wrapper_functional_proxy_99 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 287, constant_args_idx = 410, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_520, 'SUM': full_default_76, 'SUMSQ': full_default_76, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_220: "f32[512]" = triton_kernel_wrapper_functional_proxy_99['SUM']
        getitem_221: "f32[512]" = triton_kernel_wrapper_functional_proxy_99['SUMSQ'];  triton_kernel_wrapper_functional_proxy_99 = None
        div_51: "f32[512]" = torch.ops.aten.div.Tensor(getitem_220, full_default_72);  getitem_220 = None
        div_52: "f32[512]" = torch.ops.aten.div.Tensor(getitem_221, full_default_72);  getitem_221 = None
        mul_120: "f32[512]" = torch.ops.aten.mul.Tensor(div_51, div_51)
        sub_34: "f32[512]" = torch.ops.aten.sub.Tensor(div_52, mul_120);  div_52 = mul_120 = None
        clamp_min_34: "f32[512]" = torch.ops.aten.clamp_min.default(sub_34, 0.0);  sub_34 = None
        add_73: "f32[512]" = torch.ops.aten.add.Tensor(clamp_min_34, 1e-05)
        rsqrt_17: "f32[512]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        mul_121: "f32[512]" = torch.ops.aten.mul.Tensor(clamp_min_34, full_default_73);  clamp_min_34 = None
        mul_122: "f32[512]" = torch.ops.aten.mul.Tensor(primals_108, 0.9)
        mul_123: "f32[512]" = torch.ops.aten.mul.Tensor(div_51, 0.1)
        add_74: "f32[512]" = torch.ops.aten.add.Tensor(mul_122, mul_123);  mul_122 = mul_123 = None
        mul_124: "f32[512]" = torch.ops.aten.mul.Tensor(primals_109, 0.9)
        mul_125: "f32[512]" = torch.ops.aten.mul.Tensor(mul_121, 0.1);  mul_121 = None
        add_75: "f32[512]" = torch.ops.aten.add.Tensor(mul_124, mul_125);  mul_124 = mul_125 = None
        empty_154: "bf16[512, 512, 784]" = torch.ops.aten.empty.memory_format([512, 512, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_49: "bf16[512, 512, 784]" = torch.ops.aten.permute.default(empty_154, [0, 1, 2]);  empty_154 = None
        empty_155: "bf16[512, 512, 784]" = torch.ops.aten.empty.memory_format([512, 512, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_50: "bf16[512, 512, 784]" = torch.ops.aten.permute.default(empty_155, [0, 1, 2]);  empty_155 = None
        triton_kernel_wrapper_functional_proxy_100 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 288, constant_args_idx = 411, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_520, 'MEAN': div_51, 'INVSTD': rsqrt_17, 'GAMMA': primals_106, 'BETA': primals_107, 'Y': permute_49, 'X_hat': permute_50, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_520 = div_51 = primals_107 = permute_49 = permute_50 = None
        getitem_222: "bf16[512, 512, 784]" = triton_kernel_wrapper_functional_proxy_100['Y']
        getitem_223: "bf16[512, 512, 784]" = triton_kernel_wrapper_functional_proxy_100['X_hat'];  triton_kernel_wrapper_functional_proxy_100 = None
        empty_156: "i32[802816, 16]" = torch.ops.aten.empty.memory_format([802816, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_157: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_158: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_526: "bf16[512, 1568, 256]" = torch.ops.aten.view.default(getitem_223, [512, -1, 256]);  getitem_223 = None
        view_527: "bf16[802816, 256]" = torch.ops.aten.view.default(view_526, [802816, 256]);  view_526 = None
        triton_kernel_wrapper_functional_proxy_101 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 289, constant_args_idx = 412, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_527, 'P_ptr': empty_156, 'S_ptr': empty_157, 'M_ptr': empty_158, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_527 = empty_156 = empty_157 = empty_158 = None
        getitem_224: "i32[802816, 16]" = triton_kernel_wrapper_functional_proxy_101['P_ptr']
        getitem_225: "bf16[802816]" = triton_kernel_wrapper_functional_proxy_101['S_ptr']
        getitem_226: "bf16[802816]" = triton_kernel_wrapper_functional_proxy_101['M_ptr'];  triton_kernel_wrapper_functional_proxy_101 = None
        
         # File: <eval_with_key>.3 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:61 in forward, code: add_4 = layer2_1_bn3 + layer2_0_relu_2;  layer2_1_bn3 = layer2_0_relu_2 = None
        view_533: "bf16[512, 512, 28, 28]" = torch.ops.aten.view.default(getitem_222, [512, 512, 28, 28]);  getitem_222 = None
        add_76: "bf16[512, 512, 28, 28]" = torch.ops.aten.add.Tensor(view_533, getitem_188);  view_533 = getitem_188 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        empty_159: "bf16[512, 512, 28, 28]" = torch.ops.aten.empty.memory_format([512, 512, 28, 28], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_51: "bf16[512, 512, 28, 28]" = torch.ops.aten.permute.default(empty_159, [0, 1, 2, 3]);  empty_159 = None
        triton_kernel_wrapper_functional_proxy_102 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 413, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_76, 'Y_ptr': permute_51, 'Mask_prt': full_default_84, 'n_elts': 205520896, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  add_76 = permute_51 = None
        getitem_227: "bf16[512, 512, 28, 28]" = triton_kernel_wrapper_functional_proxy_102['Y_ptr']
        getitem_228: "i8[512, 512, 28, 28]" = triton_kernel_wrapper_functional_proxy_102['Mask_prt'];  triton_kernel_wrapper_functional_proxy_102 = None
        view_538: "i8[512, 1568, 256]" = torch.ops.aten.view.default(getitem_228, [512, -1, 256]);  getitem_228 = None
        view_539: "i8[802816, 256]" = torch.ops.aten.view.default(view_538, [802816, 256]);  view_538 = None
        triton_kernel_wrapper_functional_proxy_103 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 414, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_539, 'P_ptr': full_default_69, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_539 = None
        getitem_229: "i32[802816, 8]" = triton_kernel_wrapper_functional_proxy_103['P_ptr'];  triton_kernel_wrapper_functional_proxy_103 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_19: "bf16[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_110, torch.bfloat16);  primals_110 = None
        empty_160: "i32[802816, 16]" = torch.ops.aten.empty.memory_format([802816, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_161: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_162: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_544: "bf16[512, 1568, 256]" = torch.ops.aten.view.default(getitem_227, [512, -1, 256])
        view_545: "bf16[802816, 256]" = torch.ops.aten.view.default(view_544, [802816, 256]);  view_544 = None
        triton_kernel_wrapper_functional_proxy_104 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 290, constant_args_idx = 415, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_545, 'P_ptr': empty_160, 'S_ptr': empty_161, 'M_ptr': empty_162, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_545 = empty_160 = empty_161 = empty_162 = None
        getitem_230: "i32[802816, 16]" = triton_kernel_wrapper_functional_proxy_104['P_ptr']
        getitem_231: "bf16[802816]" = triton_kernel_wrapper_functional_proxy_104['S_ptr']
        getitem_232: "bf16[802816]" = triton_kernel_wrapper_functional_proxy_104['M_ptr'];  triton_kernel_wrapper_functional_proxy_104 = None
        convolution_18: "bf16[512, 128, 28, 28]" = torch.ops.aten.convolution.default(getitem_227, convert_element_type_19, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_77: "i64[]" = torch.ops.aten.add.Tensor(primals_111, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_551: "bf16[512, 128, 784]" = torch.ops.aten.view.default(convolution_18, [512, 128, 784]);  convolution_18 = None
        triton_kernel_wrapper_functional_proxy_105 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 291, constant_args_idx = 416, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_551, 'SUM': full_default_64, 'SUMSQ': full_default_64, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_233: "f32[128]" = triton_kernel_wrapper_functional_proxy_105['SUM']
        getitem_234: "f32[128]" = triton_kernel_wrapper_functional_proxy_105['SUMSQ'];  triton_kernel_wrapper_functional_proxy_105 = None
        div_54: "f32[128]" = torch.ops.aten.div.Tensor(getitem_233, full_default_72);  getitem_233 = None
        div_55: "f32[128]" = torch.ops.aten.div.Tensor(getitem_234, full_default_72);  getitem_234 = None
        mul_127: "f32[128]" = torch.ops.aten.mul.Tensor(div_54, div_54)
        sub_36: "f32[128]" = torch.ops.aten.sub.Tensor(div_55, mul_127);  div_55 = mul_127 = None
        clamp_min_36: "f32[128]" = torch.ops.aten.clamp_min.default(sub_36, 0.0);  sub_36 = None
        add_78: "f32[128]" = torch.ops.aten.add.Tensor(clamp_min_36, 1e-05)
        rsqrt_18: "f32[128]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        mul_128: "f32[128]" = torch.ops.aten.mul.Tensor(clamp_min_36, full_default_73);  clamp_min_36 = None
        mul_129: "f32[128]" = torch.ops.aten.mul.Tensor(primals_114, 0.9)
        mul_130: "f32[128]" = torch.ops.aten.mul.Tensor(div_54, 0.1)
        add_79: "f32[128]" = torch.ops.aten.add.Tensor(mul_129, mul_130);  mul_129 = mul_130 = None
        mul_131: "f32[128]" = torch.ops.aten.mul.Tensor(primals_115, 0.9)
        mul_132: "f32[128]" = torch.ops.aten.mul.Tensor(mul_128, 0.1);  mul_128 = None
        add_80: "f32[128]" = torch.ops.aten.add.Tensor(mul_131, mul_132);  mul_131 = mul_132 = None
        empty_163: "bf16[512, 128, 784]" = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_52: "bf16[512, 128, 784]" = torch.ops.aten.permute.default(empty_163, [0, 1, 2]);  empty_163 = None
        empty_164: "bf16[512, 128, 784]" = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_53: "bf16[512, 128, 784]" = torch.ops.aten.permute.default(empty_164, [0, 1, 2]);  empty_164 = None
        triton_kernel_wrapper_functional_proxy_106 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 292, constant_args_idx = 417, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_551, 'MEAN': div_54, 'INVSTD': rsqrt_18, 'GAMMA': primals_112, 'BETA': primals_113, 'Y': permute_52, 'X_hat': permute_53, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_551 = div_54 = primals_113 = permute_52 = permute_53 = None
        getitem_235: "bf16[512, 128, 784]" = triton_kernel_wrapper_functional_proxy_106['Y']
        getitem_236: "bf16[512, 128, 784]" = triton_kernel_wrapper_functional_proxy_106['X_hat'];  triton_kernel_wrapper_functional_proxy_106 = None
        empty_165: "i32[200704, 16]" = torch.ops.aten.empty.memory_format([200704, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_166: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_167: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_557: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_236, [512, -1, 256]);  getitem_236 = None
        view_558: "bf16[200704, 256]" = torch.ops.aten.view.default(view_557, [200704, 256]);  view_557 = None
        triton_kernel_wrapper_functional_proxy_107 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 293, constant_args_idx = 418, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_558, 'P_ptr': empty_165, 'S_ptr': empty_166, 'M_ptr': empty_167, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_558 = empty_165 = empty_166 = empty_167 = None
        getitem_237: "i32[200704, 16]" = triton_kernel_wrapper_functional_proxy_107['P_ptr']
        getitem_238: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_107['S_ptr']
        getitem_239: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_107['M_ptr'];  triton_kernel_wrapper_functional_proxy_107 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_564: "bf16[512, 128, 28, 28]" = torch.ops.aten.view.default(getitem_235, [512, 128, 28, 28]);  getitem_235 = None
        empty_168: "bf16[512, 128, 28, 28]" = torch.ops.aten.empty.memory_format([512, 128, 28, 28], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_54: "bf16[512, 128, 28, 28]" = torch.ops.aten.permute.default(empty_168, [0, 1, 2, 3]);  empty_168 = None
        triton_kernel_wrapper_functional_proxy_108 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 419, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_564, 'Y_ptr': permute_54, 'Mask_prt': full_default_74, 'n_elts': 51380224, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_564 = permute_54 = None
        getitem_240: "bf16[512, 128, 28, 28]" = triton_kernel_wrapper_functional_proxy_108['Y_ptr']
        getitem_241: "i8[512, 128, 28, 28]" = triton_kernel_wrapper_functional_proxy_108['Mask_prt'];  triton_kernel_wrapper_functional_proxy_108 = None
        view_569: "i8[512, 392, 256]" = torch.ops.aten.view.default(getitem_241, [512, -1, 256]);  getitem_241 = None
        view_570: "i8[200704, 256]" = torch.ops.aten.view.default(view_569, [200704, 256]);  view_569 = None
        triton_kernel_wrapper_functional_proxy_109 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 420, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_570, 'P_ptr': full_default_75, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_570 = None
        getitem_242: "i32[200704, 8]" = triton_kernel_wrapper_functional_proxy_109['P_ptr'];  triton_kernel_wrapper_functional_proxy_109 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_20: "bf16[128, 128, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_116, torch.bfloat16);  primals_116 = None
        empty_169: "i32[200704, 16]" = torch.ops.aten.empty.memory_format([200704, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_170: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_171: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_575: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_240, [512, -1, 256])
        view_576: "bf16[200704, 256]" = torch.ops.aten.view.default(view_575, [200704, 256]);  view_575 = None
        triton_kernel_wrapper_functional_proxy_110 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 294, constant_args_idx = 421, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_576, 'P_ptr': empty_169, 'S_ptr': empty_170, 'M_ptr': empty_171, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_576 = empty_169 = empty_170 = empty_171 = None
        getitem_243: "i32[200704, 16]" = triton_kernel_wrapper_functional_proxy_110['P_ptr']
        getitem_244: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_110['S_ptr']
        getitem_245: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_110['M_ptr'];  triton_kernel_wrapper_functional_proxy_110 = None
        convolution_19: "bf16[512, 128, 28, 28]" = torch.ops.aten.convolution.default(getitem_240, convert_element_type_20, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_240 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_81: "i64[]" = torch.ops.aten.add.Tensor(primals_117, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_582: "bf16[512, 128, 784]" = torch.ops.aten.view.default(convolution_19, [512, 128, 784]);  convolution_19 = None
        triton_kernel_wrapper_functional_proxy_111 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 295, constant_args_idx = 422, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_582, 'SUM': full_default_64, 'SUMSQ': full_default_64, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_246: "f32[128]" = triton_kernel_wrapper_functional_proxy_111['SUM']
        getitem_247: "f32[128]" = triton_kernel_wrapper_functional_proxy_111['SUMSQ'];  triton_kernel_wrapper_functional_proxy_111 = None
        div_57: "f32[128]" = torch.ops.aten.div.Tensor(getitem_246, full_default_72);  getitem_246 = None
        div_58: "f32[128]" = torch.ops.aten.div.Tensor(getitem_247, full_default_72);  getitem_247 = None
        mul_134: "f32[128]" = torch.ops.aten.mul.Tensor(div_57, div_57)
        sub_38: "f32[128]" = torch.ops.aten.sub.Tensor(div_58, mul_134);  div_58 = mul_134 = None
        clamp_min_38: "f32[128]" = torch.ops.aten.clamp_min.default(sub_38, 0.0);  sub_38 = None
        add_82: "f32[128]" = torch.ops.aten.add.Tensor(clamp_min_38, 1e-05)
        rsqrt_19: "f32[128]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_135: "f32[128]" = torch.ops.aten.mul.Tensor(clamp_min_38, full_default_73);  clamp_min_38 = None
        mul_136: "f32[128]" = torch.ops.aten.mul.Tensor(primals_120, 0.9)
        mul_137: "f32[128]" = torch.ops.aten.mul.Tensor(div_57, 0.1)
        add_83: "f32[128]" = torch.ops.aten.add.Tensor(mul_136, mul_137);  mul_136 = mul_137 = None
        mul_138: "f32[128]" = torch.ops.aten.mul.Tensor(primals_121, 0.9)
        mul_139: "f32[128]" = torch.ops.aten.mul.Tensor(mul_135, 0.1);  mul_135 = None
        add_84: "f32[128]" = torch.ops.aten.add.Tensor(mul_138, mul_139);  mul_138 = mul_139 = None
        empty_172: "bf16[512, 128, 784]" = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_55: "bf16[512, 128, 784]" = torch.ops.aten.permute.default(empty_172, [0, 1, 2]);  empty_172 = None
        empty_173: "bf16[512, 128, 784]" = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_56: "bf16[512, 128, 784]" = torch.ops.aten.permute.default(empty_173, [0, 1, 2]);  empty_173 = None
        triton_kernel_wrapper_functional_proxy_112 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 296, constant_args_idx = 423, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_582, 'MEAN': div_57, 'INVSTD': rsqrt_19, 'GAMMA': primals_118, 'BETA': primals_119, 'Y': permute_55, 'X_hat': permute_56, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_582 = div_57 = primals_119 = permute_55 = permute_56 = None
        getitem_248: "bf16[512, 128, 784]" = triton_kernel_wrapper_functional_proxy_112['Y']
        getitem_249: "bf16[512, 128, 784]" = triton_kernel_wrapper_functional_proxy_112['X_hat'];  triton_kernel_wrapper_functional_proxy_112 = None
        empty_174: "i32[200704, 16]" = torch.ops.aten.empty.memory_format([200704, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_175: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_176: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_588: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_249, [512, -1, 256]);  getitem_249 = None
        view_589: "bf16[200704, 256]" = torch.ops.aten.view.default(view_588, [200704, 256]);  view_588 = None
        triton_kernel_wrapper_functional_proxy_113 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 297, constant_args_idx = 424, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_589, 'P_ptr': empty_174, 'S_ptr': empty_175, 'M_ptr': empty_176, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_589 = empty_174 = empty_175 = empty_176 = None
        getitem_250: "i32[200704, 16]" = triton_kernel_wrapper_functional_proxy_113['P_ptr']
        getitem_251: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_113['S_ptr']
        getitem_252: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_113['M_ptr'];  triton_kernel_wrapper_functional_proxy_113 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_595: "bf16[512, 128, 28, 28]" = torch.ops.aten.view.default(getitem_248, [512, 128, 28, 28]);  getitem_248 = None
        empty_177: "bf16[512, 128, 28, 28]" = torch.ops.aten.empty.memory_format([512, 128, 28, 28], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_57: "bf16[512, 128, 28, 28]" = torch.ops.aten.permute.default(empty_177, [0, 1, 2, 3]);  empty_177 = None
        triton_kernel_wrapper_functional_proxy_114 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 425, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_595, 'Y_ptr': permute_57, 'Mask_prt': full_default_74, 'n_elts': 51380224, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_595 = permute_57 = None
        getitem_253: "bf16[512, 128, 28, 28]" = triton_kernel_wrapper_functional_proxy_114['Y_ptr']
        getitem_254: "i8[512, 128, 28, 28]" = triton_kernel_wrapper_functional_proxy_114['Mask_prt'];  triton_kernel_wrapper_functional_proxy_114 = None
        view_600: "i8[512, 392, 256]" = torch.ops.aten.view.default(getitem_254, [512, -1, 256]);  getitem_254 = None
        view_601: "i8[200704, 256]" = torch.ops.aten.view.default(view_600, [200704, 256]);  view_600 = None
        triton_kernel_wrapper_functional_proxy_115 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 426, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_601, 'P_ptr': full_default_75, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_601 = None
        getitem_255: "i32[200704, 8]" = triton_kernel_wrapper_functional_proxy_115['P_ptr'];  triton_kernel_wrapper_functional_proxy_115 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_21: "bf16[512, 128, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_122, torch.bfloat16);  primals_122 = None
        empty_178: "i32[200704, 16]" = torch.ops.aten.empty.memory_format([200704, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_179: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_180: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_606: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_253, [512, -1, 256])
        view_607: "bf16[200704, 256]" = torch.ops.aten.view.default(view_606, [200704, 256]);  view_606 = None
        triton_kernel_wrapper_functional_proxy_116 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 298, constant_args_idx = 427, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_607, 'P_ptr': empty_178, 'S_ptr': empty_179, 'M_ptr': empty_180, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_607 = empty_178 = empty_179 = empty_180 = None
        getitem_256: "i32[200704, 16]" = triton_kernel_wrapper_functional_proxy_116['P_ptr']
        getitem_257: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_116['S_ptr']
        getitem_258: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_116['M_ptr'];  triton_kernel_wrapper_functional_proxy_116 = None
        convolution_20: "bf16[512, 512, 28, 28]" = torch.ops.aten.convolution.default(getitem_253, convert_element_type_21, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_253 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_85: "i64[]" = torch.ops.aten.add.Tensor(primals_123, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_613: "bf16[512, 512, 784]" = torch.ops.aten.view.default(convolution_20, [512, 512, 784]);  convolution_20 = None
        triton_kernel_wrapper_functional_proxy_117 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 299, constant_args_idx = 428, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_613, 'SUM': full_default_76, 'SUMSQ': full_default_76, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_259: "f32[512]" = triton_kernel_wrapper_functional_proxy_117['SUM']
        getitem_260: "f32[512]" = triton_kernel_wrapper_functional_proxy_117['SUMSQ'];  triton_kernel_wrapper_functional_proxy_117 = None
        div_60: "f32[512]" = torch.ops.aten.div.Tensor(getitem_259, full_default_72);  getitem_259 = None
        div_61: "f32[512]" = torch.ops.aten.div.Tensor(getitem_260, full_default_72);  getitem_260 = None
        mul_141: "f32[512]" = torch.ops.aten.mul.Tensor(div_60, div_60)
        sub_40: "f32[512]" = torch.ops.aten.sub.Tensor(div_61, mul_141);  div_61 = mul_141 = None
        clamp_min_40: "f32[512]" = torch.ops.aten.clamp_min.default(sub_40, 0.0);  sub_40 = None
        add_86: "f32[512]" = torch.ops.aten.add.Tensor(clamp_min_40, 1e-05)
        rsqrt_20: "f32[512]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_142: "f32[512]" = torch.ops.aten.mul.Tensor(clamp_min_40, full_default_73);  clamp_min_40 = None
        mul_143: "f32[512]" = torch.ops.aten.mul.Tensor(primals_126, 0.9)
        mul_144: "f32[512]" = torch.ops.aten.mul.Tensor(div_60, 0.1)
        add_87: "f32[512]" = torch.ops.aten.add.Tensor(mul_143, mul_144);  mul_143 = mul_144 = None
        mul_145: "f32[512]" = torch.ops.aten.mul.Tensor(primals_127, 0.9)
        mul_146: "f32[512]" = torch.ops.aten.mul.Tensor(mul_142, 0.1);  mul_142 = None
        add_88: "f32[512]" = torch.ops.aten.add.Tensor(mul_145, mul_146);  mul_145 = mul_146 = None
        empty_181: "bf16[512, 512, 784]" = torch.ops.aten.empty.memory_format([512, 512, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_58: "bf16[512, 512, 784]" = torch.ops.aten.permute.default(empty_181, [0, 1, 2]);  empty_181 = None
        empty_182: "bf16[512, 512, 784]" = torch.ops.aten.empty.memory_format([512, 512, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_59: "bf16[512, 512, 784]" = torch.ops.aten.permute.default(empty_182, [0, 1, 2]);  empty_182 = None
        triton_kernel_wrapper_functional_proxy_118 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 300, constant_args_idx = 429, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_613, 'MEAN': div_60, 'INVSTD': rsqrt_20, 'GAMMA': primals_124, 'BETA': primals_125, 'Y': permute_58, 'X_hat': permute_59, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_613 = div_60 = primals_125 = permute_58 = permute_59 = None
        getitem_261: "bf16[512, 512, 784]" = triton_kernel_wrapper_functional_proxy_118['Y']
        getitem_262: "bf16[512, 512, 784]" = triton_kernel_wrapper_functional_proxy_118['X_hat'];  triton_kernel_wrapper_functional_proxy_118 = None
        empty_183: "i32[802816, 16]" = torch.ops.aten.empty.memory_format([802816, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_184: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_185: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_619: "bf16[512, 1568, 256]" = torch.ops.aten.view.default(getitem_262, [512, -1, 256]);  getitem_262 = None
        view_620: "bf16[802816, 256]" = torch.ops.aten.view.default(view_619, [802816, 256]);  view_619 = None
        triton_kernel_wrapper_functional_proxy_119 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 301, constant_args_idx = 430, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_620, 'P_ptr': empty_183, 'S_ptr': empty_184, 'M_ptr': empty_185, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_620 = empty_183 = empty_184 = empty_185 = None
        getitem_263: "i32[802816, 16]" = triton_kernel_wrapper_functional_proxy_119['P_ptr']
        getitem_264: "bf16[802816]" = triton_kernel_wrapper_functional_proxy_119['S_ptr']
        getitem_265: "bf16[802816]" = triton_kernel_wrapper_functional_proxy_119['M_ptr'];  triton_kernel_wrapper_functional_proxy_119 = None
        
         # File: <eval_with_key>.3 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:71 in forward, code: add_5 = layer2_2_bn3 + layer2_1_relu_2;  layer2_2_bn3 = layer2_1_relu_2 = None
        view_626: "bf16[512, 512, 28, 28]" = torch.ops.aten.view.default(getitem_261, [512, 512, 28, 28]);  getitem_261 = None
        add_89: "bf16[512, 512, 28, 28]" = torch.ops.aten.add.Tensor(view_626, getitem_227);  view_626 = getitem_227 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        empty_186: "bf16[512, 512, 28, 28]" = torch.ops.aten.empty.memory_format([512, 512, 28, 28], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_60: "bf16[512, 512, 28, 28]" = torch.ops.aten.permute.default(empty_186, [0, 1, 2, 3]);  empty_186 = None
        triton_kernel_wrapper_functional_proxy_120 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 431, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_89, 'Y_ptr': permute_60, 'Mask_prt': full_default_84, 'n_elts': 205520896, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  add_89 = permute_60 = None
        getitem_266: "bf16[512, 512, 28, 28]" = triton_kernel_wrapper_functional_proxy_120['Y_ptr']
        getitem_267: "i8[512, 512, 28, 28]" = triton_kernel_wrapper_functional_proxy_120['Mask_prt'];  triton_kernel_wrapper_functional_proxy_120 = None
        view_631: "i8[512, 1568, 256]" = torch.ops.aten.view.default(getitem_267, [512, -1, 256]);  getitem_267 = None
        view_632: "i8[802816, 256]" = torch.ops.aten.view.default(view_631, [802816, 256]);  view_631 = None
        triton_kernel_wrapper_functional_proxy_121 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 432, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_632, 'P_ptr': full_default_69, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_632 = None
        getitem_268: "i32[802816, 8]" = triton_kernel_wrapper_functional_proxy_121['P_ptr'];  triton_kernel_wrapper_functional_proxy_121 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_22: "bf16[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_128, torch.bfloat16);  primals_128 = None
        empty_187: "i32[802816, 16]" = torch.ops.aten.empty.memory_format([802816, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_188: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_189: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_637: "bf16[512, 1568, 256]" = torch.ops.aten.view.default(getitem_266, [512, -1, 256])
        view_638: "bf16[802816, 256]" = torch.ops.aten.view.default(view_637, [802816, 256]);  view_637 = None
        triton_kernel_wrapper_functional_proxy_122 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 302, constant_args_idx = 433, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_638, 'P_ptr': empty_187, 'S_ptr': empty_188, 'M_ptr': empty_189, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_638 = empty_187 = empty_188 = empty_189 = None
        getitem_269: "i32[802816, 16]" = triton_kernel_wrapper_functional_proxy_122['P_ptr']
        getitem_270: "bf16[802816]" = triton_kernel_wrapper_functional_proxy_122['S_ptr']
        getitem_271: "bf16[802816]" = triton_kernel_wrapper_functional_proxy_122['M_ptr'];  triton_kernel_wrapper_functional_proxy_122 = None
        convolution_21: "bf16[512, 128, 28, 28]" = torch.ops.aten.convolution.default(getitem_266, convert_element_type_22, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_90: "i64[]" = torch.ops.aten.add.Tensor(primals_129, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_644: "bf16[512, 128, 784]" = torch.ops.aten.view.default(convolution_21, [512, 128, 784]);  convolution_21 = None
        triton_kernel_wrapper_functional_proxy_123 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 303, constant_args_idx = 434, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_644, 'SUM': full_default_64, 'SUMSQ': full_default_64, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_272: "f32[128]" = triton_kernel_wrapper_functional_proxy_123['SUM']
        getitem_273: "f32[128]" = triton_kernel_wrapper_functional_proxy_123['SUMSQ'];  triton_kernel_wrapper_functional_proxy_123 = None
        div_63: "f32[128]" = torch.ops.aten.div.Tensor(getitem_272, full_default_72);  getitem_272 = None
        div_64: "f32[128]" = torch.ops.aten.div.Tensor(getitem_273, full_default_72);  getitem_273 = None
        mul_148: "f32[128]" = torch.ops.aten.mul.Tensor(div_63, div_63)
        sub_42: "f32[128]" = torch.ops.aten.sub.Tensor(div_64, mul_148);  div_64 = mul_148 = None
        clamp_min_42: "f32[128]" = torch.ops.aten.clamp_min.default(sub_42, 0.0);  sub_42 = None
        add_91: "f32[128]" = torch.ops.aten.add.Tensor(clamp_min_42, 1e-05)
        rsqrt_21: "f32[128]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
        mul_149: "f32[128]" = torch.ops.aten.mul.Tensor(clamp_min_42, full_default_73);  clamp_min_42 = None
        mul_150: "f32[128]" = torch.ops.aten.mul.Tensor(primals_132, 0.9)
        mul_151: "f32[128]" = torch.ops.aten.mul.Tensor(div_63, 0.1)
        add_92: "f32[128]" = torch.ops.aten.add.Tensor(mul_150, mul_151);  mul_150 = mul_151 = None
        mul_152: "f32[128]" = torch.ops.aten.mul.Tensor(primals_133, 0.9)
        mul_153: "f32[128]" = torch.ops.aten.mul.Tensor(mul_149, 0.1);  mul_149 = None
        add_93: "f32[128]" = torch.ops.aten.add.Tensor(mul_152, mul_153);  mul_152 = mul_153 = None
        empty_190: "bf16[512, 128, 784]" = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_61: "bf16[512, 128, 784]" = torch.ops.aten.permute.default(empty_190, [0, 1, 2]);  empty_190 = None
        empty_191: "bf16[512, 128, 784]" = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_62: "bf16[512, 128, 784]" = torch.ops.aten.permute.default(empty_191, [0, 1, 2]);  empty_191 = None
        triton_kernel_wrapper_functional_proxy_124 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 304, constant_args_idx = 435, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_644, 'MEAN': div_63, 'INVSTD': rsqrt_21, 'GAMMA': primals_130, 'BETA': primals_131, 'Y': permute_61, 'X_hat': permute_62, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_644 = div_63 = primals_131 = permute_61 = permute_62 = None
        getitem_274: "bf16[512, 128, 784]" = triton_kernel_wrapper_functional_proxy_124['Y']
        getitem_275: "bf16[512, 128, 784]" = triton_kernel_wrapper_functional_proxy_124['X_hat'];  triton_kernel_wrapper_functional_proxy_124 = None
        empty_192: "i32[200704, 16]" = torch.ops.aten.empty.memory_format([200704, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_193: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_194: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_650: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_275, [512, -1, 256]);  getitem_275 = None
        view_651: "bf16[200704, 256]" = torch.ops.aten.view.default(view_650, [200704, 256]);  view_650 = None
        triton_kernel_wrapper_functional_proxy_125 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 305, constant_args_idx = 436, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_651, 'P_ptr': empty_192, 'S_ptr': empty_193, 'M_ptr': empty_194, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_651 = empty_192 = empty_193 = empty_194 = None
        getitem_276: "i32[200704, 16]" = triton_kernel_wrapper_functional_proxy_125['P_ptr']
        getitem_277: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_125['S_ptr']
        getitem_278: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_125['M_ptr'];  triton_kernel_wrapper_functional_proxy_125 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_657: "bf16[512, 128, 28, 28]" = torch.ops.aten.view.default(getitem_274, [512, 128, 28, 28]);  getitem_274 = None
        empty_195: "bf16[512, 128, 28, 28]" = torch.ops.aten.empty.memory_format([512, 128, 28, 28], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_63: "bf16[512, 128, 28, 28]" = torch.ops.aten.permute.default(empty_195, [0, 1, 2, 3]);  empty_195 = None
        triton_kernel_wrapper_functional_proxy_126 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 437, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_657, 'Y_ptr': permute_63, 'Mask_prt': full_default_74, 'n_elts': 51380224, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_657 = permute_63 = None
        getitem_279: "bf16[512, 128, 28, 28]" = triton_kernel_wrapper_functional_proxy_126['Y_ptr']
        getitem_280: "i8[512, 128, 28, 28]" = triton_kernel_wrapper_functional_proxy_126['Mask_prt'];  triton_kernel_wrapper_functional_proxy_126 = None
        view_662: "i8[512, 392, 256]" = torch.ops.aten.view.default(getitem_280, [512, -1, 256]);  getitem_280 = None
        view_663: "i8[200704, 256]" = torch.ops.aten.view.default(view_662, [200704, 256]);  view_662 = None
        triton_kernel_wrapper_functional_proxy_127 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 438, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_663, 'P_ptr': full_default_75, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_663 = None
        getitem_281: "i32[200704, 8]" = triton_kernel_wrapper_functional_proxy_127['P_ptr'];  triton_kernel_wrapper_functional_proxy_127 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_23: "bf16[128, 128, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_134, torch.bfloat16);  primals_134 = None
        empty_196: "i32[200704, 16]" = torch.ops.aten.empty.memory_format([200704, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_197: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_198: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_668: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_279, [512, -1, 256])
        view_669: "bf16[200704, 256]" = torch.ops.aten.view.default(view_668, [200704, 256]);  view_668 = None
        triton_kernel_wrapper_functional_proxy_128 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 306, constant_args_idx = 439, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_669, 'P_ptr': empty_196, 'S_ptr': empty_197, 'M_ptr': empty_198, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_669 = empty_196 = empty_197 = empty_198 = None
        getitem_282: "i32[200704, 16]" = triton_kernel_wrapper_functional_proxy_128['P_ptr']
        getitem_283: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_128['S_ptr']
        getitem_284: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_128['M_ptr'];  triton_kernel_wrapper_functional_proxy_128 = None
        convolution_22: "bf16[512, 128, 28, 28]" = torch.ops.aten.convolution.default(getitem_279, convert_element_type_23, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_279 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_94: "i64[]" = torch.ops.aten.add.Tensor(primals_135, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_675: "bf16[512, 128, 784]" = torch.ops.aten.view.default(convolution_22, [512, 128, 784]);  convolution_22 = None
        triton_kernel_wrapper_functional_proxy_129 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 307, constant_args_idx = 440, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_675, 'SUM': full_default_64, 'SUMSQ': full_default_64, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ']);  full_default_64 = None
        getitem_285: "f32[128]" = triton_kernel_wrapper_functional_proxy_129['SUM']
        getitem_286: "f32[128]" = triton_kernel_wrapper_functional_proxy_129['SUMSQ'];  triton_kernel_wrapper_functional_proxy_129 = None
        div_66: "f32[128]" = torch.ops.aten.div.Tensor(getitem_285, full_default_72);  getitem_285 = None
        div_67: "f32[128]" = torch.ops.aten.div.Tensor(getitem_286, full_default_72);  getitem_286 = None
        mul_155: "f32[128]" = torch.ops.aten.mul.Tensor(div_66, div_66)
        sub_44: "f32[128]" = torch.ops.aten.sub.Tensor(div_67, mul_155);  div_67 = mul_155 = None
        clamp_min_44: "f32[128]" = torch.ops.aten.clamp_min.default(sub_44, 0.0);  sub_44 = None
        add_95: "f32[128]" = torch.ops.aten.add.Tensor(clamp_min_44, 1e-05)
        rsqrt_22: "f32[128]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
        mul_156: "f32[128]" = torch.ops.aten.mul.Tensor(clamp_min_44, full_default_73);  clamp_min_44 = None
        mul_157: "f32[128]" = torch.ops.aten.mul.Tensor(primals_138, 0.9)
        mul_158: "f32[128]" = torch.ops.aten.mul.Tensor(div_66, 0.1)
        add_96: "f32[128]" = torch.ops.aten.add.Tensor(mul_157, mul_158);  mul_157 = mul_158 = None
        mul_159: "f32[128]" = torch.ops.aten.mul.Tensor(primals_139, 0.9)
        mul_160: "f32[128]" = torch.ops.aten.mul.Tensor(mul_156, 0.1);  mul_156 = None
        add_97: "f32[128]" = torch.ops.aten.add.Tensor(mul_159, mul_160);  mul_159 = mul_160 = None
        empty_199: "bf16[512, 128, 784]" = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_64: "bf16[512, 128, 784]" = torch.ops.aten.permute.default(empty_199, [0, 1, 2]);  empty_199 = None
        empty_200: "bf16[512, 128, 784]" = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_65: "bf16[512, 128, 784]" = torch.ops.aten.permute.default(empty_200, [0, 1, 2]);  empty_200 = None
        triton_kernel_wrapper_functional_proxy_130 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 308, constant_args_idx = 441, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_675, 'MEAN': div_66, 'INVSTD': rsqrt_22, 'GAMMA': primals_136, 'BETA': primals_137, 'Y': permute_64, 'X_hat': permute_65, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_675 = div_66 = primals_137 = permute_64 = permute_65 = None
        getitem_287: "bf16[512, 128, 784]" = triton_kernel_wrapper_functional_proxy_130['Y']
        getitem_288: "bf16[512, 128, 784]" = triton_kernel_wrapper_functional_proxy_130['X_hat'];  triton_kernel_wrapper_functional_proxy_130 = None
        empty_201: "i32[200704, 16]" = torch.ops.aten.empty.memory_format([200704, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_202: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_203: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_681: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_288, [512, -1, 256]);  getitem_288 = None
        view_682: "bf16[200704, 256]" = torch.ops.aten.view.default(view_681, [200704, 256]);  view_681 = None
        triton_kernel_wrapper_functional_proxy_131 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 309, constant_args_idx = 442, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_682, 'P_ptr': empty_201, 'S_ptr': empty_202, 'M_ptr': empty_203, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_682 = empty_201 = empty_202 = empty_203 = None
        getitem_289: "i32[200704, 16]" = triton_kernel_wrapper_functional_proxy_131['P_ptr']
        getitem_290: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_131['S_ptr']
        getitem_291: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_131['M_ptr'];  triton_kernel_wrapper_functional_proxy_131 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_688: "bf16[512, 128, 28, 28]" = torch.ops.aten.view.default(getitem_287, [512, 128, 28, 28]);  getitem_287 = None
        empty_204: "bf16[512, 128, 28, 28]" = torch.ops.aten.empty.memory_format([512, 128, 28, 28], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_66: "bf16[512, 128, 28, 28]" = torch.ops.aten.permute.default(empty_204, [0, 1, 2, 3]);  empty_204 = None
        triton_kernel_wrapper_functional_proxy_132 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 443, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_688, 'Y_ptr': permute_66, 'Mask_prt': full_default_74, 'n_elts': 51380224, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_688 = permute_66 = full_default_74 = None
        getitem_292: "bf16[512, 128, 28, 28]" = triton_kernel_wrapper_functional_proxy_132['Y_ptr']
        getitem_293: "i8[512, 128, 28, 28]" = triton_kernel_wrapper_functional_proxy_132['Mask_prt'];  triton_kernel_wrapper_functional_proxy_132 = None
        view_693: "i8[512, 392, 256]" = torch.ops.aten.view.default(getitem_293, [512, -1, 256]);  getitem_293 = None
        view_694: "i8[200704, 256]" = torch.ops.aten.view.default(view_693, [200704, 256]);  view_693 = None
        triton_kernel_wrapper_functional_proxy_133 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 444, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_694, 'P_ptr': full_default_75, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_694 = None
        getitem_294: "i32[200704, 8]" = triton_kernel_wrapper_functional_proxy_133['P_ptr'];  triton_kernel_wrapper_functional_proxy_133 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_24: "bf16[512, 128, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_140, torch.bfloat16);  primals_140 = None
        empty_205: "i32[200704, 16]" = torch.ops.aten.empty.memory_format([200704, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_206: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_207: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_699: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_292, [512, -1, 256])
        view_700: "bf16[200704, 256]" = torch.ops.aten.view.default(view_699, [200704, 256]);  view_699 = None
        triton_kernel_wrapper_functional_proxy_134 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 310, constant_args_idx = 445, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_700, 'P_ptr': empty_205, 'S_ptr': empty_206, 'M_ptr': empty_207, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_700 = empty_205 = empty_206 = empty_207 = None
        getitem_295: "i32[200704, 16]" = triton_kernel_wrapper_functional_proxy_134['P_ptr']
        getitem_296: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_134['S_ptr']
        getitem_297: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_134['M_ptr'];  triton_kernel_wrapper_functional_proxy_134 = None
        convolution_23: "bf16[512, 512, 28, 28]" = torch.ops.aten.convolution.default(getitem_292, convert_element_type_24, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_292 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_98: "i64[]" = torch.ops.aten.add.Tensor(primals_141, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_706: "bf16[512, 512, 784]" = torch.ops.aten.view.default(convolution_23, [512, 512, 784]);  convolution_23 = None
        triton_kernel_wrapper_functional_proxy_135 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 311, constant_args_idx = 446, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_706, 'SUM': full_default_76, 'SUMSQ': full_default_76, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_298: "f32[512]" = triton_kernel_wrapper_functional_proxy_135['SUM']
        getitem_299: "f32[512]" = triton_kernel_wrapper_functional_proxy_135['SUMSQ'];  triton_kernel_wrapper_functional_proxy_135 = None
        div_69: "f32[512]" = torch.ops.aten.div.Tensor(getitem_298, full_default_72);  getitem_298 = None
        div_70: "f32[512]" = torch.ops.aten.div.Tensor(getitem_299, full_default_72);  getitem_299 = None
        mul_162: "f32[512]" = torch.ops.aten.mul.Tensor(div_69, div_69)
        sub_46: "f32[512]" = torch.ops.aten.sub.Tensor(div_70, mul_162);  div_70 = mul_162 = None
        clamp_min_46: "f32[512]" = torch.ops.aten.clamp_min.default(sub_46, 0.0);  sub_46 = None
        add_99: "f32[512]" = torch.ops.aten.add.Tensor(clamp_min_46, 1e-05)
        rsqrt_23: "f32[512]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        mul_163: "f32[512]" = torch.ops.aten.mul.Tensor(clamp_min_46, full_default_73);  clamp_min_46 = None
        mul_164: "f32[512]" = torch.ops.aten.mul.Tensor(primals_144, 0.9)
        mul_165: "f32[512]" = torch.ops.aten.mul.Tensor(div_69, 0.1)
        add_100: "f32[512]" = torch.ops.aten.add.Tensor(mul_164, mul_165);  mul_164 = mul_165 = None
        mul_166: "f32[512]" = torch.ops.aten.mul.Tensor(primals_145, 0.9)
        mul_167: "f32[512]" = torch.ops.aten.mul.Tensor(mul_163, 0.1);  mul_163 = None
        add_101: "f32[512]" = torch.ops.aten.add.Tensor(mul_166, mul_167);  mul_166 = mul_167 = None
        empty_208: "bf16[512, 512, 784]" = torch.ops.aten.empty.memory_format([512, 512, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_67: "bf16[512, 512, 784]" = torch.ops.aten.permute.default(empty_208, [0, 1, 2]);  empty_208 = None
        empty_209: "bf16[512, 512, 784]" = torch.ops.aten.empty.memory_format([512, 512, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_68: "bf16[512, 512, 784]" = torch.ops.aten.permute.default(empty_209, [0, 1, 2]);  empty_209 = None
        triton_kernel_wrapper_functional_proxy_136 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 312, constant_args_idx = 447, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_706, 'MEAN': div_69, 'INVSTD': rsqrt_23, 'GAMMA': primals_142, 'BETA': primals_143, 'Y': permute_67, 'X_hat': permute_68, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_706 = div_69 = primals_143 = permute_67 = permute_68 = None
        getitem_300: "bf16[512, 512, 784]" = triton_kernel_wrapper_functional_proxy_136['Y']
        getitem_301: "bf16[512, 512, 784]" = triton_kernel_wrapper_functional_proxy_136['X_hat'];  triton_kernel_wrapper_functional_proxy_136 = None
        empty_210: "i32[802816, 16]" = torch.ops.aten.empty.memory_format([802816, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_211: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_212: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_712: "bf16[512, 1568, 256]" = torch.ops.aten.view.default(getitem_301, [512, -1, 256]);  getitem_301 = None
        view_713: "bf16[802816, 256]" = torch.ops.aten.view.default(view_712, [802816, 256]);  view_712 = None
        triton_kernel_wrapper_functional_proxy_137 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 313, constant_args_idx = 448, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_713, 'P_ptr': empty_210, 'S_ptr': empty_211, 'M_ptr': empty_212, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_713 = empty_210 = empty_211 = empty_212 = None
        getitem_302: "i32[802816, 16]" = triton_kernel_wrapper_functional_proxy_137['P_ptr']
        getitem_303: "bf16[802816]" = triton_kernel_wrapper_functional_proxy_137['S_ptr']
        getitem_304: "bf16[802816]" = triton_kernel_wrapper_functional_proxy_137['M_ptr'];  triton_kernel_wrapper_functional_proxy_137 = None
        
         # File: <eval_with_key>.3 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:81 in forward, code: add_6 = layer2_3_bn3 + layer2_2_relu_2;  layer2_3_bn3 = layer2_2_relu_2 = None
        view_719: "bf16[512, 512, 28, 28]" = torch.ops.aten.view.default(getitem_300, [512, 512, 28, 28]);  getitem_300 = None
        add_102: "bf16[512, 512, 28, 28]" = torch.ops.aten.add.Tensor(view_719, getitem_266);  view_719 = getitem_266 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        empty_213: "bf16[512, 512, 28, 28]" = torch.ops.aten.empty.memory_format([512, 512, 28, 28], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_69: "bf16[512, 512, 28, 28]" = torch.ops.aten.permute.default(empty_213, [0, 1, 2, 3]);  empty_213 = None
        triton_kernel_wrapper_functional_proxy_138 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 449, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_102, 'Y_ptr': permute_69, 'Mask_prt': full_default_84, 'n_elts': 205520896, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  add_102 = permute_69 = full_default_84 = None
        getitem_305: "bf16[512, 512, 28, 28]" = triton_kernel_wrapper_functional_proxy_138['Y_ptr']
        getitem_306: "i8[512, 512, 28, 28]" = triton_kernel_wrapper_functional_proxy_138['Mask_prt'];  triton_kernel_wrapper_functional_proxy_138 = None
        view_724: "i8[512, 1568, 256]" = torch.ops.aten.view.default(getitem_306, [512, -1, 256]);  getitem_306 = None
        view_725: "i8[802816, 256]" = torch.ops.aten.view.default(view_724, [802816, 256]);  view_724 = None
        triton_kernel_wrapper_functional_proxy_139 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 450, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_725, 'P_ptr': full_default_69, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_725 = full_default_69 = None
        getitem_307: "i32[802816, 8]" = triton_kernel_wrapper_functional_proxy_139['P_ptr'];  triton_kernel_wrapper_functional_proxy_139 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_25: "bf16[256, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_146, torch.bfloat16);  primals_146 = None
        empty_214: "i32[802816, 16]" = torch.ops.aten.empty.memory_format([802816, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_215: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_216: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_730: "bf16[512, 1568, 256]" = torch.ops.aten.view.default(getitem_305, [512, -1, 256])
        view_731: "bf16[802816, 256]" = torch.ops.aten.view.default(view_730, [802816, 256]);  view_730 = None
        triton_kernel_wrapper_functional_proxy_140 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 314, constant_args_idx = 451, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_731, 'P_ptr': empty_214, 'S_ptr': empty_215, 'M_ptr': empty_216, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  empty_214 = empty_215 = empty_216 = None
        getitem_308: "i32[802816, 16]" = triton_kernel_wrapper_functional_proxy_140['P_ptr']
        getitem_309: "bf16[802816]" = triton_kernel_wrapper_functional_proxy_140['S_ptr']
        getitem_310: "bf16[802816]" = triton_kernel_wrapper_functional_proxy_140['M_ptr'];  triton_kernel_wrapper_functional_proxy_140 = None
        convolution_24: "bf16[512, 256, 28, 28]" = torch.ops.aten.convolution.default(getitem_305, convert_element_type_25, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_103: "i64[]" = torch.ops.aten.add.Tensor(primals_147, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_737: "bf16[512, 256, 784]" = torch.ops.aten.view.default(convolution_24, [512, 256, 784]);  convolution_24 = None
        triton_kernel_wrapper_functional_proxy_141 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 315, constant_args_idx = 452, grid = [(256, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_737, 'SUM': full_default_18, 'SUMSQ': full_default_18, 'M': 401408, 'HW': 784, 'stride_n': 200704, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_311: "f32[256]" = triton_kernel_wrapper_functional_proxy_141['SUM']
        getitem_312: "f32[256]" = triton_kernel_wrapper_functional_proxy_141['SUMSQ'];  triton_kernel_wrapper_functional_proxy_141 = None
        div_72: "f32[256]" = torch.ops.aten.div.Tensor(getitem_311, full_default_72);  getitem_311 = None
        div_73: "f32[256]" = torch.ops.aten.div.Tensor(getitem_312, full_default_72);  getitem_312 = full_default_72 = None
        mul_169: "f32[256]" = torch.ops.aten.mul.Tensor(div_72, div_72)
        sub_48: "f32[256]" = torch.ops.aten.sub.Tensor(div_73, mul_169);  div_73 = mul_169 = None
        clamp_min_48: "f32[256]" = torch.ops.aten.clamp_min.default(sub_48, 0.0);  sub_48 = None
        add_104: "f32[256]" = torch.ops.aten.add.Tensor(clamp_min_48, 1e-05)
        rsqrt_24: "f32[256]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        mul_170: "f32[256]" = torch.ops.aten.mul.Tensor(clamp_min_48, full_default_73);  clamp_min_48 = full_default_73 = None
        mul_171: "f32[256]" = torch.ops.aten.mul.Tensor(primals_150, 0.9)
        mul_172: "f32[256]" = torch.ops.aten.mul.Tensor(div_72, 0.1)
        add_105: "f32[256]" = torch.ops.aten.add.Tensor(mul_171, mul_172);  mul_171 = mul_172 = None
        mul_173: "f32[256]" = torch.ops.aten.mul.Tensor(primals_151, 0.9)
        mul_174: "f32[256]" = torch.ops.aten.mul.Tensor(mul_170, 0.1);  mul_170 = None
        add_106: "f32[256]" = torch.ops.aten.add.Tensor(mul_173, mul_174);  mul_173 = mul_174 = None
        empty_217: "bf16[512, 256, 784]" = torch.ops.aten.empty.memory_format([512, 256, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_70: "bf16[512, 256, 784]" = torch.ops.aten.permute.default(empty_217, [0, 1, 2]);  empty_217 = None
        empty_218: "bf16[512, 256, 784]" = torch.ops.aten.empty.memory_format([512, 256, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_71: "bf16[512, 256, 784]" = torch.ops.aten.permute.default(empty_218, [0, 1, 2]);  empty_218 = None
        triton_kernel_wrapper_functional_proxy_142 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 316, constant_args_idx = 453, grid = [(256, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_737, 'MEAN': div_72, 'INVSTD': rsqrt_24, 'GAMMA': primals_148, 'BETA': primals_149, 'Y': permute_70, 'X_hat': permute_71, 'M': 401408, 'HW': 784, 'stride_n': 200704, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_737 = div_72 = primals_149 = permute_70 = permute_71 = None
        getitem_313: "bf16[512, 256, 784]" = triton_kernel_wrapper_functional_proxy_142['Y']
        getitem_314: "bf16[512, 256, 784]" = triton_kernel_wrapper_functional_proxy_142['X_hat'];  triton_kernel_wrapper_functional_proxy_142 = None
        empty_219: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_220: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_221: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_743: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_314, [512, -1, 256]);  getitem_314 = None
        view_744: "bf16[401408, 256]" = torch.ops.aten.view.default(view_743, [401408, 256]);  view_743 = None
        triton_kernel_wrapper_functional_proxy_143 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 317, constant_args_idx = 454, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_744, 'P_ptr': empty_219, 'S_ptr': empty_220, 'M_ptr': empty_221, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_744 = empty_219 = empty_220 = empty_221 = None
        getitem_315: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_143['P_ptr']
        getitem_316: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_143['S_ptr']
        getitem_317: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_143['M_ptr'];  triton_kernel_wrapper_functional_proxy_143 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_144: "i8[512, 256, 28, 28]" = torch.ops.aten.full.default([512, 256, 28, 28], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_750: "bf16[512, 256, 28, 28]" = torch.ops.aten.view.default(getitem_313, [512, 256, 28, 28]);  getitem_313 = None
        empty_222: "bf16[512, 256, 28, 28]" = torch.ops.aten.empty.memory_format([512, 256, 28, 28], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_72: "bf16[512, 256, 28, 28]" = torch.ops.aten.permute.default(empty_222, [0, 1, 2, 3]);  empty_222 = None
        triton_kernel_wrapper_functional_proxy_144 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 455, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_750, 'Y_ptr': permute_72, 'Mask_prt': full_default_144, 'n_elts': 102760448, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_750 = permute_72 = full_default_144 = None
        getitem_318: "bf16[512, 256, 28, 28]" = triton_kernel_wrapper_functional_proxy_144['Y_ptr']
        getitem_319: "i8[512, 256, 28, 28]" = triton_kernel_wrapper_functional_proxy_144['Mask_prt'];  triton_kernel_wrapper_functional_proxy_144 = None
        view_755: "i8[512, 784, 256]" = torch.ops.aten.view.default(getitem_319, [512, -1, 256]);  getitem_319 = None
        view_756: "i8[401408, 256]" = torch.ops.aten.view.default(view_755, [401408, 256]);  view_755 = None
        triton_kernel_wrapper_functional_proxy_145 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 456, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_756, 'P_ptr': full_default_11, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_756 = None
        getitem_320: "i32[401408, 8]" = triton_kernel_wrapper_functional_proxy_145['P_ptr'];  triton_kernel_wrapper_functional_proxy_145 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_26: "bf16[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_152, torch.bfloat16);  primals_152 = None
        empty_223: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_224: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_225: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_761: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_318, [512, -1, 256])
        view_762: "bf16[401408, 256]" = torch.ops.aten.view.default(view_761, [401408, 256]);  view_761 = None
        triton_kernel_wrapper_functional_proxy_146 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 318, constant_args_idx = 457, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_762, 'P_ptr': empty_223, 'S_ptr': empty_224, 'M_ptr': empty_225, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_762 = empty_223 = empty_224 = empty_225 = None
        getitem_321: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_146['P_ptr']
        getitem_322: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_146['S_ptr']
        getitem_323: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_146['M_ptr'];  triton_kernel_wrapper_functional_proxy_146 = None
        convolution_25: "bf16[512, 256, 14, 14]" = torch.ops.aten.convolution.default(getitem_318, convert_element_type_26, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_318 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_107: "i64[]" = torch.ops.aten.add.Tensor(primals_153, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_768: "bf16[512, 256, 196]" = torch.ops.aten.view.default(convolution_25, [512, 256, 196]);  convolution_25 = None
        triton_kernel_wrapper_functional_proxy_147 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 319, constant_args_idx = 458, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_768, 'SUM': full_default_18, 'SUMSQ': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_324: "f32[256]" = triton_kernel_wrapper_functional_proxy_147['SUM']
        getitem_325: "f32[256]" = triton_kernel_wrapper_functional_proxy_147['SUMSQ'];  triton_kernel_wrapper_functional_proxy_147 = None
        full_default_148: "f32[]" = torch.ops.aten.full.default([], 100352.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        div_75: "f32[256]" = torch.ops.aten.div.Tensor(getitem_324, full_default_148);  getitem_324 = None
        div_76: "f32[256]" = torch.ops.aten.div.Tensor(getitem_325, full_default_148);  getitem_325 = None
        mul_176: "f32[256]" = torch.ops.aten.mul.Tensor(div_75, div_75)
        sub_50: "f32[256]" = torch.ops.aten.sub.Tensor(div_76, mul_176);  div_76 = mul_176 = None
        clamp_min_50: "f32[256]" = torch.ops.aten.clamp_min.default(sub_50, 0.0);  sub_50 = None
        add_108: "f32[256]" = torch.ops.aten.add.Tensor(clamp_min_50, 1e-05)
        rsqrt_25: "f32[256]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        full_default_149: "f32[]" = torch.ops.aten.full.default([], 1.0000100135803223, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        mul_177: "f32[256]" = torch.ops.aten.mul.Tensor(clamp_min_50, full_default_149);  clamp_min_50 = None
        mul_178: "f32[256]" = torch.ops.aten.mul.Tensor(primals_156, 0.9)
        mul_179: "f32[256]" = torch.ops.aten.mul.Tensor(div_75, 0.1)
        add_109: "f32[256]" = torch.ops.aten.add.Tensor(mul_178, mul_179);  mul_178 = mul_179 = None
        mul_180: "f32[256]" = torch.ops.aten.mul.Tensor(primals_157, 0.9)
        mul_181: "f32[256]" = torch.ops.aten.mul.Tensor(mul_177, 0.1);  mul_177 = None
        add_110: "f32[256]" = torch.ops.aten.add.Tensor(mul_180, mul_181);  mul_180 = mul_181 = None
        empty_226: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_73: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_226, [0, 1, 2]);  empty_226 = None
        empty_227: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_74: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_227, [0, 1, 2]);  empty_227 = None
        triton_kernel_wrapper_functional_proxy_148 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 320, constant_args_idx = 459, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_768, 'MEAN': div_75, 'INVSTD': rsqrt_25, 'GAMMA': primals_154, 'BETA': primals_155, 'Y': permute_73, 'X_hat': permute_74, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_768 = div_75 = primals_155 = permute_73 = permute_74 = None
        getitem_326: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_148['Y']
        getitem_327: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_148['X_hat'];  triton_kernel_wrapper_functional_proxy_148 = None
        empty_228: "i32[100352, 16]" = torch.ops.aten.empty.memory_format([100352, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_229: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_230: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_774: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_327, [512, -1, 256]);  getitem_327 = None
        view_775: "bf16[100352, 256]" = torch.ops.aten.view.default(view_774, [100352, 256]);  view_774 = None
        triton_kernel_wrapper_functional_proxy_149 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 321, constant_args_idx = 460, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_775, 'P_ptr': empty_228, 'S_ptr': empty_229, 'M_ptr': empty_230, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_775 = empty_228 = empty_229 = empty_230 = None
        getitem_328: "i32[100352, 16]" = triton_kernel_wrapper_functional_proxy_149['P_ptr']
        getitem_329: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_149['S_ptr']
        getitem_330: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_149['M_ptr'];  triton_kernel_wrapper_functional_proxy_149 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_150: "i8[512, 256, 14, 14]" = torch.ops.aten.full.default([512, 256, 14, 14], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_781: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(getitem_326, [512, 256, 14, 14]);  getitem_326 = None
        empty_231: "bf16[512, 256, 14, 14]" = torch.ops.aten.empty.memory_format([512, 256, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_75: "bf16[512, 256, 14, 14]" = torch.ops.aten.permute.default(empty_231, [0, 1, 2, 3]);  empty_231 = None
        triton_kernel_wrapper_functional_proxy_150 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 461, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_781, 'Y_ptr': permute_75, 'Mask_prt': full_default_150, 'n_elts': 25690112, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_781 = permute_75 = None
        getitem_331: "bf16[512, 256, 14, 14]" = triton_kernel_wrapper_functional_proxy_150['Y_ptr']
        getitem_332: "i8[512, 256, 14, 14]" = triton_kernel_wrapper_functional_proxy_150['Mask_prt'];  triton_kernel_wrapper_functional_proxy_150 = None
        full_default_151: "i32[100352, 8]" = torch.ops.aten.full.default([100352, 8], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_786: "i8[512, 196, 256]" = torch.ops.aten.view.default(getitem_332, [512, -1, 256]);  getitem_332 = None
        view_787: "i8[100352, 256]" = torch.ops.aten.view.default(view_786, [100352, 256]);  view_786 = None
        triton_kernel_wrapper_functional_proxy_151 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 462, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_787, 'P_ptr': full_default_151, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_787 = None
        getitem_333: "i32[100352, 8]" = triton_kernel_wrapper_functional_proxy_151['P_ptr'];  triton_kernel_wrapper_functional_proxy_151 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_27: "bf16[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_158, torch.bfloat16);  primals_158 = None
        empty_232: "i32[100352, 16]" = torch.ops.aten.empty.memory_format([100352, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_233: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_234: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_792: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_331, [512, -1, 256])
        view_793: "bf16[100352, 256]" = torch.ops.aten.view.default(view_792, [100352, 256]);  view_792 = None
        triton_kernel_wrapper_functional_proxy_152 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 322, constant_args_idx = 463, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_793, 'P_ptr': empty_232, 'S_ptr': empty_233, 'M_ptr': empty_234, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_793 = empty_232 = empty_233 = empty_234 = None
        getitem_334: "i32[100352, 16]" = triton_kernel_wrapper_functional_proxy_152['P_ptr']
        getitem_335: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_152['S_ptr']
        getitem_336: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_152['M_ptr'];  triton_kernel_wrapper_functional_proxy_152 = None
        convolution_26: "bf16[512, 1024, 14, 14]" = torch.ops.aten.convolution.default(getitem_331, convert_element_type_27, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_331 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_111: "i64[]" = torch.ops.aten.add.Tensor(primals_159, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_799: "bf16[512, 1024, 196]" = torch.ops.aten.view.default(convolution_26, [512, 1024, 196]);  convolution_26 = None
        full_default_152: "f32[1024]" = torch.ops.aten.full.default([1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_153 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 323, constant_args_idx = 464, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_799, 'SUM': full_default_152, 'SUMSQ': full_default_152, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_337: "f32[1024]" = triton_kernel_wrapper_functional_proxy_153['SUM']
        getitem_338: "f32[1024]" = triton_kernel_wrapper_functional_proxy_153['SUMSQ'];  triton_kernel_wrapper_functional_proxy_153 = None
        div_78: "f32[1024]" = torch.ops.aten.div.Tensor(getitem_337, full_default_148);  getitem_337 = None
        div_79: "f32[1024]" = torch.ops.aten.div.Tensor(getitem_338, full_default_148);  getitem_338 = None
        mul_183: "f32[1024]" = torch.ops.aten.mul.Tensor(div_78, div_78)
        sub_52: "f32[1024]" = torch.ops.aten.sub.Tensor(div_79, mul_183);  div_79 = mul_183 = None
        clamp_min_52: "f32[1024]" = torch.ops.aten.clamp_min.default(sub_52, 0.0);  sub_52 = None
        add_112: "f32[1024]" = torch.ops.aten.add.Tensor(clamp_min_52, 1e-05)
        rsqrt_26: "f32[1024]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        mul_184: "f32[1024]" = torch.ops.aten.mul.Tensor(clamp_min_52, full_default_149);  clamp_min_52 = None
        mul_185: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_162, 0.9)
        mul_186: "f32[1024]" = torch.ops.aten.mul.Tensor(div_78, 0.1)
        add_113: "f32[1024]" = torch.ops.aten.add.Tensor(mul_185, mul_186);  mul_185 = mul_186 = None
        mul_187: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_163, 0.9)
        mul_188: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_184, 0.1);  mul_184 = None
        add_114: "f32[1024]" = torch.ops.aten.add.Tensor(mul_187, mul_188);  mul_187 = mul_188 = None
        empty_235: "bf16[512, 1024, 196]" = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_76: "bf16[512, 1024, 196]" = torch.ops.aten.permute.default(empty_235, [0, 1, 2]);  empty_235 = None
        empty_236: "bf16[512, 1024, 196]" = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_77: "bf16[512, 1024, 196]" = torch.ops.aten.permute.default(empty_236, [0, 1, 2]);  empty_236 = None
        triton_kernel_wrapper_functional_proxy_154 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 324, constant_args_idx = 465, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_799, 'MEAN': div_78, 'INVSTD': rsqrt_26, 'GAMMA': primals_160, 'BETA': primals_161, 'Y': permute_76, 'X_hat': permute_77, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_799 = div_78 = primals_161 = permute_76 = permute_77 = None
        getitem_339: "bf16[512, 1024, 196]" = triton_kernel_wrapper_functional_proxy_154['Y']
        getitem_340: "bf16[512, 1024, 196]" = triton_kernel_wrapper_functional_proxy_154['X_hat'];  triton_kernel_wrapper_functional_proxy_154 = None
        empty_237: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_238: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_239: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_805: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_340, [512, -1, 256]);  getitem_340 = None
        view_806: "bf16[401408, 256]" = torch.ops.aten.view.default(view_805, [401408, 256]);  view_805 = None
        triton_kernel_wrapper_functional_proxy_155 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 325, constant_args_idx = 466, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_806, 'P_ptr': empty_237, 'S_ptr': empty_238, 'M_ptr': empty_239, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_806 = empty_237 = empty_238 = empty_239 = None
        getitem_341: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_155['P_ptr']
        getitem_342: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_155['S_ptr']
        getitem_343: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_155['M_ptr'];  triton_kernel_wrapper_functional_proxy_155 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_28: "bf16[1024, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_164, torch.bfloat16);  primals_164 = None
        empty_240: "i32[802816, 16]" = torch.ops.aten.empty.memory_format([802816, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_241: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_242: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_156 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 326, constant_args_idx = 467, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_731, 'P_ptr': empty_240, 'S_ptr': empty_241, 'M_ptr': empty_242, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_731 = empty_240 = empty_241 = empty_242 = None
        getitem_344: "i32[802816, 16]" = triton_kernel_wrapper_functional_proxy_156['P_ptr']
        getitem_345: "bf16[802816]" = triton_kernel_wrapper_functional_proxy_156['S_ptr']
        getitem_346: "bf16[802816]" = triton_kernel_wrapper_functional_proxy_156['M_ptr'];  triton_kernel_wrapper_functional_proxy_156 = None
        convolution_27: "bf16[512, 1024, 14, 14]" = torch.ops.aten.convolution.default(getitem_305, convert_element_type_28, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  getitem_305 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_115: "i64[]" = torch.ops.aten.add.Tensor(primals_165, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_823: "bf16[512, 1024, 196]" = torch.ops.aten.view.default(convolution_27, [512, 1024, 196]);  convolution_27 = None
        triton_kernel_wrapper_functional_proxy_157 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 327, constant_args_idx = 468, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_823, 'SUM': full_default_152, 'SUMSQ': full_default_152, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_347: "f32[1024]" = triton_kernel_wrapper_functional_proxy_157['SUM']
        getitem_348: "f32[1024]" = triton_kernel_wrapper_functional_proxy_157['SUMSQ'];  triton_kernel_wrapper_functional_proxy_157 = None
        div_81: "f32[1024]" = torch.ops.aten.div.Tensor(getitem_347, full_default_148);  getitem_347 = None
        div_82: "f32[1024]" = torch.ops.aten.div.Tensor(getitem_348, full_default_148);  getitem_348 = None
        mul_190: "f32[1024]" = torch.ops.aten.mul.Tensor(div_81, div_81)
        sub_54: "f32[1024]" = torch.ops.aten.sub.Tensor(div_82, mul_190);  div_82 = mul_190 = None
        clamp_min_54: "f32[1024]" = torch.ops.aten.clamp_min.default(sub_54, 0.0);  sub_54 = None
        add_116: "f32[1024]" = torch.ops.aten.add.Tensor(clamp_min_54, 1e-05)
        rsqrt_27: "f32[1024]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        mul_191: "f32[1024]" = torch.ops.aten.mul.Tensor(clamp_min_54, full_default_149);  clamp_min_54 = None
        mul_192: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_168, 0.9)
        mul_193: "f32[1024]" = torch.ops.aten.mul.Tensor(div_81, 0.1)
        add_117: "f32[1024]" = torch.ops.aten.add.Tensor(mul_192, mul_193);  mul_192 = mul_193 = None
        mul_194: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_169, 0.9)
        mul_195: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_191, 0.1);  mul_191 = None
        add_118: "f32[1024]" = torch.ops.aten.add.Tensor(mul_194, mul_195);  mul_194 = mul_195 = None
        empty_243: "bf16[512, 1024, 196]" = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_78: "bf16[512, 1024, 196]" = torch.ops.aten.permute.default(empty_243, [0, 1, 2]);  empty_243 = None
        empty_244: "bf16[512, 1024, 196]" = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_79: "bf16[512, 1024, 196]" = torch.ops.aten.permute.default(empty_244, [0, 1, 2]);  empty_244 = None
        triton_kernel_wrapper_functional_proxy_158 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 328, constant_args_idx = 469, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_823, 'MEAN': div_81, 'INVSTD': rsqrt_27, 'GAMMA': primals_166, 'BETA': primals_167, 'Y': permute_78, 'X_hat': permute_79, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_823 = div_81 = primals_167 = permute_78 = permute_79 = None
        getitem_349: "bf16[512, 1024, 196]" = triton_kernel_wrapper_functional_proxy_158['Y']
        getitem_350: "bf16[512, 1024, 196]" = triton_kernel_wrapper_functional_proxy_158['X_hat'];  triton_kernel_wrapper_functional_proxy_158 = None
        empty_245: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_246: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_247: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_829: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_350, [512, -1, 256]);  getitem_350 = None
        view_830: "bf16[401408, 256]" = torch.ops.aten.view.default(view_829, [401408, 256]);  view_829 = None
        triton_kernel_wrapper_functional_proxy_159 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 329, constant_args_idx = 470, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_830, 'P_ptr': empty_245, 'S_ptr': empty_246, 'M_ptr': empty_247, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_830 = empty_245 = empty_246 = empty_247 = None
        getitem_351: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_159['P_ptr']
        getitem_352: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_159['S_ptr']
        getitem_353: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_159['M_ptr'];  triton_kernel_wrapper_functional_proxy_159 = None
        
         # File: <eval_with_key>.3 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:93 in forward, code: add_7 = layer3_0_bn3 + layer3_0_downsample_1;  layer3_0_bn3 = layer3_0_downsample_1 = None
        view_836: "bf16[512, 1024, 14, 14]" = torch.ops.aten.view.default(getitem_339, [512, 1024, 14, 14]);  getitem_339 = None
        view_837: "bf16[512, 1024, 14, 14]" = torch.ops.aten.view.default(getitem_349, [512, 1024, 14, 14]);  getitem_349 = None
        add_119: "bf16[512, 1024, 14, 14]" = torch.ops.aten.add.Tensor(view_836, view_837);  view_836 = view_837 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_160: "i8[512, 1024, 14, 14]" = torch.ops.aten.full.default([512, 1024, 14, 14], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        empty_248: "bf16[512, 1024, 14, 14]" = torch.ops.aten.empty.memory_format([512, 1024, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_80: "bf16[512, 1024, 14, 14]" = torch.ops.aten.permute.default(empty_248, [0, 1, 2, 3]);  empty_248 = None
        triton_kernel_wrapper_functional_proxy_160 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 471, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_119, 'Y_ptr': permute_80, 'Mask_prt': full_default_160, 'n_elts': 102760448, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  add_119 = permute_80 = None
        getitem_354: "bf16[512, 1024, 14, 14]" = triton_kernel_wrapper_functional_proxy_160['Y_ptr']
        getitem_355: "i8[512, 1024, 14, 14]" = triton_kernel_wrapper_functional_proxy_160['Mask_prt'];  triton_kernel_wrapper_functional_proxy_160 = None
        view_842: "i8[512, 784, 256]" = torch.ops.aten.view.default(getitem_355, [512, -1, 256]);  getitem_355 = None
        view_843: "i8[401408, 256]" = torch.ops.aten.view.default(view_842, [401408, 256]);  view_842 = None
        triton_kernel_wrapper_functional_proxy_161 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 472, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_843, 'P_ptr': full_default_11, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_843 = None
        getitem_356: "i32[401408, 8]" = triton_kernel_wrapper_functional_proxy_161['P_ptr'];  triton_kernel_wrapper_functional_proxy_161 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_29: "bf16[256, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_170, torch.bfloat16);  primals_170 = None
        empty_249: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_250: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_251: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_848: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_354, [512, -1, 256])
        view_849: "bf16[401408, 256]" = torch.ops.aten.view.default(view_848, [401408, 256]);  view_848 = None
        triton_kernel_wrapper_functional_proxy_162 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 330, constant_args_idx = 473, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_849, 'P_ptr': empty_249, 'S_ptr': empty_250, 'M_ptr': empty_251, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_849 = empty_249 = empty_250 = empty_251 = None
        getitem_357: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_162['P_ptr']
        getitem_358: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_162['S_ptr']
        getitem_359: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_162['M_ptr'];  triton_kernel_wrapper_functional_proxy_162 = None
        convolution_28: "bf16[512, 256, 14, 14]" = torch.ops.aten.convolution.default(getitem_354, convert_element_type_29, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_120: "i64[]" = torch.ops.aten.add.Tensor(primals_171, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_855: "bf16[512, 256, 196]" = torch.ops.aten.view.default(convolution_28, [512, 256, 196]);  convolution_28 = None
        triton_kernel_wrapper_functional_proxy_163 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 331, constant_args_idx = 474, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_855, 'SUM': full_default_18, 'SUMSQ': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_360: "f32[256]" = triton_kernel_wrapper_functional_proxy_163['SUM']
        getitem_361: "f32[256]" = triton_kernel_wrapper_functional_proxy_163['SUMSQ'];  triton_kernel_wrapper_functional_proxy_163 = None
        div_84: "f32[256]" = torch.ops.aten.div.Tensor(getitem_360, full_default_148);  getitem_360 = None
        div_85: "f32[256]" = torch.ops.aten.div.Tensor(getitem_361, full_default_148);  getitem_361 = None
        mul_197: "f32[256]" = torch.ops.aten.mul.Tensor(div_84, div_84)
        sub_56: "f32[256]" = torch.ops.aten.sub.Tensor(div_85, mul_197);  div_85 = mul_197 = None
        clamp_min_56: "f32[256]" = torch.ops.aten.clamp_min.default(sub_56, 0.0);  sub_56 = None
        add_121: "f32[256]" = torch.ops.aten.add.Tensor(clamp_min_56, 1e-05)
        rsqrt_28: "f32[256]" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
        mul_198: "f32[256]" = torch.ops.aten.mul.Tensor(clamp_min_56, full_default_149);  clamp_min_56 = None
        mul_199: "f32[256]" = torch.ops.aten.mul.Tensor(primals_174, 0.9)
        mul_200: "f32[256]" = torch.ops.aten.mul.Tensor(div_84, 0.1)
        add_122: "f32[256]" = torch.ops.aten.add.Tensor(mul_199, mul_200);  mul_199 = mul_200 = None
        mul_201: "f32[256]" = torch.ops.aten.mul.Tensor(primals_175, 0.9)
        mul_202: "f32[256]" = torch.ops.aten.mul.Tensor(mul_198, 0.1);  mul_198 = None
        add_123: "f32[256]" = torch.ops.aten.add.Tensor(mul_201, mul_202);  mul_201 = mul_202 = None
        empty_252: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_81: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_252, [0, 1, 2]);  empty_252 = None
        empty_253: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_82: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_253, [0, 1, 2]);  empty_253 = None
        triton_kernel_wrapper_functional_proxy_164 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 332, constant_args_idx = 475, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_855, 'MEAN': div_84, 'INVSTD': rsqrt_28, 'GAMMA': primals_172, 'BETA': primals_173, 'Y': permute_81, 'X_hat': permute_82, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_855 = div_84 = primals_173 = permute_81 = permute_82 = None
        getitem_362: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_164['Y']
        getitem_363: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_164['X_hat'];  triton_kernel_wrapper_functional_proxy_164 = None
        empty_254: "i32[100352, 16]" = torch.ops.aten.empty.memory_format([100352, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_255: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_256: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_861: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_363, [512, -1, 256]);  getitem_363 = None
        view_862: "bf16[100352, 256]" = torch.ops.aten.view.default(view_861, [100352, 256]);  view_861 = None
        triton_kernel_wrapper_functional_proxy_165 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 333, constant_args_idx = 476, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_862, 'P_ptr': empty_254, 'S_ptr': empty_255, 'M_ptr': empty_256, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_862 = empty_254 = empty_255 = empty_256 = None
        getitem_364: "i32[100352, 16]" = triton_kernel_wrapper_functional_proxy_165['P_ptr']
        getitem_365: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_165['S_ptr']
        getitem_366: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_165['M_ptr'];  triton_kernel_wrapper_functional_proxy_165 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_868: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(getitem_362, [512, 256, 14, 14]);  getitem_362 = None
        empty_257: "bf16[512, 256, 14, 14]" = torch.ops.aten.empty.memory_format([512, 256, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_83: "bf16[512, 256, 14, 14]" = torch.ops.aten.permute.default(empty_257, [0, 1, 2, 3]);  empty_257 = None
        triton_kernel_wrapper_functional_proxy_166 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 477, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_868, 'Y_ptr': permute_83, 'Mask_prt': full_default_150, 'n_elts': 25690112, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_868 = permute_83 = None
        getitem_367: "bf16[512, 256, 14, 14]" = triton_kernel_wrapper_functional_proxy_166['Y_ptr']
        getitem_368: "i8[512, 256, 14, 14]" = triton_kernel_wrapper_functional_proxy_166['Mask_prt'];  triton_kernel_wrapper_functional_proxy_166 = None
        view_873: "i8[512, 196, 256]" = torch.ops.aten.view.default(getitem_368, [512, -1, 256]);  getitem_368 = None
        view_874: "i8[100352, 256]" = torch.ops.aten.view.default(view_873, [100352, 256]);  view_873 = None
        triton_kernel_wrapper_functional_proxy_167 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 478, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_874, 'P_ptr': full_default_151, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_874 = None
        getitem_369: "i32[100352, 8]" = triton_kernel_wrapper_functional_proxy_167['P_ptr'];  triton_kernel_wrapper_functional_proxy_167 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_30: "bf16[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_176, torch.bfloat16);  primals_176 = None
        empty_258: "i32[100352, 16]" = torch.ops.aten.empty.memory_format([100352, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_259: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_260: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_879: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_367, [512, -1, 256])
        view_880: "bf16[100352, 256]" = torch.ops.aten.view.default(view_879, [100352, 256]);  view_879 = None
        triton_kernel_wrapper_functional_proxy_168 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 334, constant_args_idx = 479, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_880, 'P_ptr': empty_258, 'S_ptr': empty_259, 'M_ptr': empty_260, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_880 = empty_258 = empty_259 = empty_260 = None
        getitem_370: "i32[100352, 16]" = triton_kernel_wrapper_functional_proxy_168['P_ptr']
        getitem_371: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_168['S_ptr']
        getitem_372: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_168['M_ptr'];  triton_kernel_wrapper_functional_proxy_168 = None
        convolution_29: "bf16[512, 256, 14, 14]" = torch.ops.aten.convolution.default(getitem_367, convert_element_type_30, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_367 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_124: "i64[]" = torch.ops.aten.add.Tensor(primals_177, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_886: "bf16[512, 256, 196]" = torch.ops.aten.view.default(convolution_29, [512, 256, 196]);  convolution_29 = None
        triton_kernel_wrapper_functional_proxy_169 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 335, constant_args_idx = 480, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_886, 'SUM': full_default_18, 'SUMSQ': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_373: "f32[256]" = triton_kernel_wrapper_functional_proxy_169['SUM']
        getitem_374: "f32[256]" = triton_kernel_wrapper_functional_proxy_169['SUMSQ'];  triton_kernel_wrapper_functional_proxy_169 = None
        div_87: "f32[256]" = torch.ops.aten.div.Tensor(getitem_373, full_default_148);  getitem_373 = None
        div_88: "f32[256]" = torch.ops.aten.div.Tensor(getitem_374, full_default_148);  getitem_374 = None
        mul_204: "f32[256]" = torch.ops.aten.mul.Tensor(div_87, div_87)
        sub_58: "f32[256]" = torch.ops.aten.sub.Tensor(div_88, mul_204);  div_88 = mul_204 = None
        clamp_min_58: "f32[256]" = torch.ops.aten.clamp_min.default(sub_58, 0.0);  sub_58 = None
        add_125: "f32[256]" = torch.ops.aten.add.Tensor(clamp_min_58, 1e-05)
        rsqrt_29: "f32[256]" = torch.ops.aten.rsqrt.default(add_125);  add_125 = None
        mul_205: "f32[256]" = torch.ops.aten.mul.Tensor(clamp_min_58, full_default_149);  clamp_min_58 = None
        mul_206: "f32[256]" = torch.ops.aten.mul.Tensor(primals_180, 0.9)
        mul_207: "f32[256]" = torch.ops.aten.mul.Tensor(div_87, 0.1)
        add_126: "f32[256]" = torch.ops.aten.add.Tensor(mul_206, mul_207);  mul_206 = mul_207 = None
        mul_208: "f32[256]" = torch.ops.aten.mul.Tensor(primals_181, 0.9)
        mul_209: "f32[256]" = torch.ops.aten.mul.Tensor(mul_205, 0.1);  mul_205 = None
        add_127: "f32[256]" = torch.ops.aten.add.Tensor(mul_208, mul_209);  mul_208 = mul_209 = None
        empty_261: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_84: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_261, [0, 1, 2]);  empty_261 = None
        empty_262: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_85: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_262, [0, 1, 2]);  empty_262 = None
        triton_kernel_wrapper_functional_proxy_170 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 336, constant_args_idx = 481, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_886, 'MEAN': div_87, 'INVSTD': rsqrt_29, 'GAMMA': primals_178, 'BETA': primals_179, 'Y': permute_84, 'X_hat': permute_85, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_886 = div_87 = primals_179 = permute_84 = permute_85 = None
        getitem_375: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_170['Y']
        getitem_376: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_170['X_hat'];  triton_kernel_wrapper_functional_proxy_170 = None
        empty_263: "i32[100352, 16]" = torch.ops.aten.empty.memory_format([100352, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_264: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_265: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_892: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_376, [512, -1, 256]);  getitem_376 = None
        view_893: "bf16[100352, 256]" = torch.ops.aten.view.default(view_892, [100352, 256]);  view_892 = None
        triton_kernel_wrapper_functional_proxy_171 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 337, constant_args_idx = 482, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_893, 'P_ptr': empty_263, 'S_ptr': empty_264, 'M_ptr': empty_265, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_893 = empty_263 = empty_264 = empty_265 = None
        getitem_377: "i32[100352, 16]" = triton_kernel_wrapper_functional_proxy_171['P_ptr']
        getitem_378: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_171['S_ptr']
        getitem_379: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_171['M_ptr'];  triton_kernel_wrapper_functional_proxy_171 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_899: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(getitem_375, [512, 256, 14, 14]);  getitem_375 = None
        empty_266: "bf16[512, 256, 14, 14]" = torch.ops.aten.empty.memory_format([512, 256, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_86: "bf16[512, 256, 14, 14]" = torch.ops.aten.permute.default(empty_266, [0, 1, 2, 3]);  empty_266 = None
        triton_kernel_wrapper_functional_proxy_172 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 483, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_899, 'Y_ptr': permute_86, 'Mask_prt': full_default_150, 'n_elts': 25690112, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_899 = permute_86 = None
        getitem_380: "bf16[512, 256, 14, 14]" = triton_kernel_wrapper_functional_proxy_172['Y_ptr']
        getitem_381: "i8[512, 256, 14, 14]" = triton_kernel_wrapper_functional_proxy_172['Mask_prt'];  triton_kernel_wrapper_functional_proxy_172 = None
        view_904: "i8[512, 196, 256]" = torch.ops.aten.view.default(getitem_381, [512, -1, 256]);  getitem_381 = None
        view_905: "i8[100352, 256]" = torch.ops.aten.view.default(view_904, [100352, 256]);  view_904 = None
        triton_kernel_wrapper_functional_proxy_173 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 484, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_905, 'P_ptr': full_default_151, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_905 = None
        getitem_382: "i32[100352, 8]" = triton_kernel_wrapper_functional_proxy_173['P_ptr'];  triton_kernel_wrapper_functional_proxy_173 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_31: "bf16[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_182, torch.bfloat16);  primals_182 = None
        empty_267: "i32[100352, 16]" = torch.ops.aten.empty.memory_format([100352, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_268: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_269: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_910: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_380, [512, -1, 256])
        view_911: "bf16[100352, 256]" = torch.ops.aten.view.default(view_910, [100352, 256]);  view_910 = None
        triton_kernel_wrapper_functional_proxy_174 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 338, constant_args_idx = 485, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_911, 'P_ptr': empty_267, 'S_ptr': empty_268, 'M_ptr': empty_269, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_911 = empty_267 = empty_268 = empty_269 = None
        getitem_383: "i32[100352, 16]" = triton_kernel_wrapper_functional_proxy_174['P_ptr']
        getitem_384: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_174['S_ptr']
        getitem_385: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_174['M_ptr'];  triton_kernel_wrapper_functional_proxy_174 = None
        convolution_30: "bf16[512, 1024, 14, 14]" = torch.ops.aten.convolution.default(getitem_380, convert_element_type_31, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_380 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_128: "i64[]" = torch.ops.aten.add.Tensor(primals_183, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_917: "bf16[512, 1024, 196]" = torch.ops.aten.view.default(convolution_30, [512, 1024, 196]);  convolution_30 = None
        triton_kernel_wrapper_functional_proxy_175 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 339, constant_args_idx = 486, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_917, 'SUM': full_default_152, 'SUMSQ': full_default_152, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_386: "f32[1024]" = triton_kernel_wrapper_functional_proxy_175['SUM']
        getitem_387: "f32[1024]" = triton_kernel_wrapper_functional_proxy_175['SUMSQ'];  triton_kernel_wrapper_functional_proxy_175 = None
        div_90: "f32[1024]" = torch.ops.aten.div.Tensor(getitem_386, full_default_148);  getitem_386 = None
        div_91: "f32[1024]" = torch.ops.aten.div.Tensor(getitem_387, full_default_148);  getitem_387 = None
        mul_211: "f32[1024]" = torch.ops.aten.mul.Tensor(div_90, div_90)
        sub_60: "f32[1024]" = torch.ops.aten.sub.Tensor(div_91, mul_211);  div_91 = mul_211 = None
        clamp_min_60: "f32[1024]" = torch.ops.aten.clamp_min.default(sub_60, 0.0);  sub_60 = None
        add_129: "f32[1024]" = torch.ops.aten.add.Tensor(clamp_min_60, 1e-05)
        rsqrt_30: "f32[1024]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        mul_212: "f32[1024]" = torch.ops.aten.mul.Tensor(clamp_min_60, full_default_149);  clamp_min_60 = None
        mul_213: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_186, 0.9)
        mul_214: "f32[1024]" = torch.ops.aten.mul.Tensor(div_90, 0.1)
        add_130: "f32[1024]" = torch.ops.aten.add.Tensor(mul_213, mul_214);  mul_213 = mul_214 = None
        mul_215: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_187, 0.9)
        mul_216: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_212, 0.1);  mul_212 = None
        add_131: "f32[1024]" = torch.ops.aten.add.Tensor(mul_215, mul_216);  mul_215 = mul_216 = None
        empty_270: "bf16[512, 1024, 196]" = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_87: "bf16[512, 1024, 196]" = torch.ops.aten.permute.default(empty_270, [0, 1, 2]);  empty_270 = None
        empty_271: "bf16[512, 1024, 196]" = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_88: "bf16[512, 1024, 196]" = torch.ops.aten.permute.default(empty_271, [0, 1, 2]);  empty_271 = None
        triton_kernel_wrapper_functional_proxy_176 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 340, constant_args_idx = 487, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_917, 'MEAN': div_90, 'INVSTD': rsqrt_30, 'GAMMA': primals_184, 'BETA': primals_185, 'Y': permute_87, 'X_hat': permute_88, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_917 = div_90 = primals_185 = permute_87 = permute_88 = None
        getitem_388: "bf16[512, 1024, 196]" = triton_kernel_wrapper_functional_proxy_176['Y']
        getitem_389: "bf16[512, 1024, 196]" = triton_kernel_wrapper_functional_proxy_176['X_hat'];  triton_kernel_wrapper_functional_proxy_176 = None
        empty_272: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_273: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_274: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_923: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_389, [512, -1, 256]);  getitem_389 = None
        view_924: "bf16[401408, 256]" = torch.ops.aten.view.default(view_923, [401408, 256]);  view_923 = None
        triton_kernel_wrapper_functional_proxy_177 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 341, constant_args_idx = 488, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_924, 'P_ptr': empty_272, 'S_ptr': empty_273, 'M_ptr': empty_274, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_924 = empty_272 = empty_273 = empty_274 = None
        getitem_390: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_177['P_ptr']
        getitem_391: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_177['S_ptr']
        getitem_392: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_177['M_ptr'];  triton_kernel_wrapper_functional_proxy_177 = None
        
         # File: <eval_with_key>.3 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:103 in forward, code: add_8 = layer3_1_bn3 + layer3_0_relu_2;  layer3_1_bn3 = layer3_0_relu_2 = None
        view_930: "bf16[512, 1024, 14, 14]" = torch.ops.aten.view.default(getitem_388, [512, 1024, 14, 14]);  getitem_388 = None
        add_132: "bf16[512, 1024, 14, 14]" = torch.ops.aten.add.Tensor(view_930, getitem_354);  view_930 = getitem_354 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        empty_275: "bf16[512, 1024, 14, 14]" = torch.ops.aten.empty.memory_format([512, 1024, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_89: "bf16[512, 1024, 14, 14]" = torch.ops.aten.permute.default(empty_275, [0, 1, 2, 3]);  empty_275 = None
        triton_kernel_wrapper_functional_proxy_178 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 489, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_132, 'Y_ptr': permute_89, 'Mask_prt': full_default_160, 'n_elts': 102760448, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  add_132 = permute_89 = None
        getitem_393: "bf16[512, 1024, 14, 14]" = triton_kernel_wrapper_functional_proxy_178['Y_ptr']
        getitem_394: "i8[512, 1024, 14, 14]" = triton_kernel_wrapper_functional_proxy_178['Mask_prt'];  triton_kernel_wrapper_functional_proxy_178 = None
        view_935: "i8[512, 784, 256]" = torch.ops.aten.view.default(getitem_394, [512, -1, 256]);  getitem_394 = None
        view_936: "i8[401408, 256]" = torch.ops.aten.view.default(view_935, [401408, 256]);  view_935 = None
        triton_kernel_wrapper_functional_proxy_179 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 490, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_936, 'P_ptr': full_default_11, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_936 = None
        getitem_395: "i32[401408, 8]" = triton_kernel_wrapper_functional_proxy_179['P_ptr'];  triton_kernel_wrapper_functional_proxy_179 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_32: "bf16[256, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_188, torch.bfloat16);  primals_188 = None
        empty_276: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_277: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_278: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_941: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_393, [512, -1, 256])
        view_942: "bf16[401408, 256]" = torch.ops.aten.view.default(view_941, [401408, 256]);  view_941 = None
        triton_kernel_wrapper_functional_proxy_180 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 342, constant_args_idx = 491, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_942, 'P_ptr': empty_276, 'S_ptr': empty_277, 'M_ptr': empty_278, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_942 = empty_276 = empty_277 = empty_278 = None
        getitem_396: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_180['P_ptr']
        getitem_397: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_180['S_ptr']
        getitem_398: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_180['M_ptr'];  triton_kernel_wrapper_functional_proxy_180 = None
        convolution_31: "bf16[512, 256, 14, 14]" = torch.ops.aten.convolution.default(getitem_393, convert_element_type_32, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_133: "i64[]" = torch.ops.aten.add.Tensor(primals_189, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_948: "bf16[512, 256, 196]" = torch.ops.aten.view.default(convolution_31, [512, 256, 196]);  convolution_31 = None
        triton_kernel_wrapper_functional_proxy_181 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 343, constant_args_idx = 492, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_948, 'SUM': full_default_18, 'SUMSQ': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_399: "f32[256]" = triton_kernel_wrapper_functional_proxy_181['SUM']
        getitem_400: "f32[256]" = triton_kernel_wrapper_functional_proxy_181['SUMSQ'];  triton_kernel_wrapper_functional_proxy_181 = None
        div_93: "f32[256]" = torch.ops.aten.div.Tensor(getitem_399, full_default_148);  getitem_399 = None
        div_94: "f32[256]" = torch.ops.aten.div.Tensor(getitem_400, full_default_148);  getitem_400 = None
        mul_218: "f32[256]" = torch.ops.aten.mul.Tensor(div_93, div_93)
        sub_62: "f32[256]" = torch.ops.aten.sub.Tensor(div_94, mul_218);  div_94 = mul_218 = None
        clamp_min_62: "f32[256]" = torch.ops.aten.clamp_min.default(sub_62, 0.0);  sub_62 = None
        add_134: "f32[256]" = torch.ops.aten.add.Tensor(clamp_min_62, 1e-05)
        rsqrt_31: "f32[256]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        mul_219: "f32[256]" = torch.ops.aten.mul.Tensor(clamp_min_62, full_default_149);  clamp_min_62 = None
        mul_220: "f32[256]" = torch.ops.aten.mul.Tensor(primals_192, 0.9)
        mul_221: "f32[256]" = torch.ops.aten.mul.Tensor(div_93, 0.1)
        add_135: "f32[256]" = torch.ops.aten.add.Tensor(mul_220, mul_221);  mul_220 = mul_221 = None
        mul_222: "f32[256]" = torch.ops.aten.mul.Tensor(primals_193, 0.9)
        mul_223: "f32[256]" = torch.ops.aten.mul.Tensor(mul_219, 0.1);  mul_219 = None
        add_136: "f32[256]" = torch.ops.aten.add.Tensor(mul_222, mul_223);  mul_222 = mul_223 = None
        empty_279: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_90: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_279, [0, 1, 2]);  empty_279 = None
        empty_280: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_91: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_280, [0, 1, 2]);  empty_280 = None
        triton_kernel_wrapper_functional_proxy_182 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 344, constant_args_idx = 493, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_948, 'MEAN': div_93, 'INVSTD': rsqrt_31, 'GAMMA': primals_190, 'BETA': primals_191, 'Y': permute_90, 'X_hat': permute_91, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_948 = div_93 = primals_191 = permute_90 = permute_91 = None
        getitem_401: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_182['Y']
        getitem_402: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_182['X_hat'];  triton_kernel_wrapper_functional_proxy_182 = None
        empty_281: "i32[100352, 16]" = torch.ops.aten.empty.memory_format([100352, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_282: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_283: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_954: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_402, [512, -1, 256]);  getitem_402 = None
        view_955: "bf16[100352, 256]" = torch.ops.aten.view.default(view_954, [100352, 256]);  view_954 = None
        triton_kernel_wrapper_functional_proxy_183 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 345, constant_args_idx = 494, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_955, 'P_ptr': empty_281, 'S_ptr': empty_282, 'M_ptr': empty_283, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_955 = empty_281 = empty_282 = empty_283 = None
        getitem_403: "i32[100352, 16]" = triton_kernel_wrapper_functional_proxy_183['P_ptr']
        getitem_404: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_183['S_ptr']
        getitem_405: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_183['M_ptr'];  triton_kernel_wrapper_functional_proxy_183 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_961: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(getitem_401, [512, 256, 14, 14]);  getitem_401 = None
        empty_284: "bf16[512, 256, 14, 14]" = torch.ops.aten.empty.memory_format([512, 256, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_92: "bf16[512, 256, 14, 14]" = torch.ops.aten.permute.default(empty_284, [0, 1, 2, 3]);  empty_284 = None
        triton_kernel_wrapper_functional_proxy_184 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 495, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_961, 'Y_ptr': permute_92, 'Mask_prt': full_default_150, 'n_elts': 25690112, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_961 = permute_92 = None
        getitem_406: "bf16[512, 256, 14, 14]" = triton_kernel_wrapper_functional_proxy_184['Y_ptr']
        getitem_407: "i8[512, 256, 14, 14]" = triton_kernel_wrapper_functional_proxy_184['Mask_prt'];  triton_kernel_wrapper_functional_proxy_184 = None
        view_966: "i8[512, 196, 256]" = torch.ops.aten.view.default(getitem_407, [512, -1, 256]);  getitem_407 = None
        view_967: "i8[100352, 256]" = torch.ops.aten.view.default(view_966, [100352, 256]);  view_966 = None
        triton_kernel_wrapper_functional_proxy_185 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 496, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_967, 'P_ptr': full_default_151, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_967 = None
        getitem_408: "i32[100352, 8]" = triton_kernel_wrapper_functional_proxy_185['P_ptr'];  triton_kernel_wrapper_functional_proxy_185 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_33: "bf16[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_194, torch.bfloat16);  primals_194 = None
        empty_285: "i32[100352, 16]" = torch.ops.aten.empty.memory_format([100352, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_286: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_287: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_972: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_406, [512, -1, 256])
        view_973: "bf16[100352, 256]" = torch.ops.aten.view.default(view_972, [100352, 256]);  view_972 = None
        triton_kernel_wrapper_functional_proxy_186 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 346, constant_args_idx = 497, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_973, 'P_ptr': empty_285, 'S_ptr': empty_286, 'M_ptr': empty_287, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_973 = empty_285 = empty_286 = empty_287 = None
        getitem_409: "i32[100352, 16]" = triton_kernel_wrapper_functional_proxy_186['P_ptr']
        getitem_410: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_186['S_ptr']
        getitem_411: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_186['M_ptr'];  triton_kernel_wrapper_functional_proxy_186 = None
        convolution_32: "bf16[512, 256, 14, 14]" = torch.ops.aten.convolution.default(getitem_406, convert_element_type_33, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_406 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_137: "i64[]" = torch.ops.aten.add.Tensor(primals_195, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_979: "bf16[512, 256, 196]" = torch.ops.aten.view.default(convolution_32, [512, 256, 196]);  convolution_32 = None
        triton_kernel_wrapper_functional_proxy_187 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 347, constant_args_idx = 498, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_979, 'SUM': full_default_18, 'SUMSQ': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_412: "f32[256]" = triton_kernel_wrapper_functional_proxy_187['SUM']
        getitem_413: "f32[256]" = triton_kernel_wrapper_functional_proxy_187['SUMSQ'];  triton_kernel_wrapper_functional_proxy_187 = None
        div_96: "f32[256]" = torch.ops.aten.div.Tensor(getitem_412, full_default_148);  getitem_412 = None
        div_97: "f32[256]" = torch.ops.aten.div.Tensor(getitem_413, full_default_148);  getitem_413 = None
        mul_225: "f32[256]" = torch.ops.aten.mul.Tensor(div_96, div_96)
        sub_64: "f32[256]" = torch.ops.aten.sub.Tensor(div_97, mul_225);  div_97 = mul_225 = None
        clamp_min_64: "f32[256]" = torch.ops.aten.clamp_min.default(sub_64, 0.0);  sub_64 = None
        add_138: "f32[256]" = torch.ops.aten.add.Tensor(clamp_min_64, 1e-05)
        rsqrt_32: "f32[256]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        mul_226: "f32[256]" = torch.ops.aten.mul.Tensor(clamp_min_64, full_default_149);  clamp_min_64 = None
        mul_227: "f32[256]" = torch.ops.aten.mul.Tensor(primals_198, 0.9)
        mul_228: "f32[256]" = torch.ops.aten.mul.Tensor(div_96, 0.1)
        add_139: "f32[256]" = torch.ops.aten.add.Tensor(mul_227, mul_228);  mul_227 = mul_228 = None
        mul_229: "f32[256]" = torch.ops.aten.mul.Tensor(primals_199, 0.9)
        mul_230: "f32[256]" = torch.ops.aten.mul.Tensor(mul_226, 0.1);  mul_226 = None
        add_140: "f32[256]" = torch.ops.aten.add.Tensor(mul_229, mul_230);  mul_229 = mul_230 = None
        empty_288: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_93: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_288, [0, 1, 2]);  empty_288 = None
        empty_289: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_94: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_289, [0, 1, 2]);  empty_289 = None
        triton_kernel_wrapper_functional_proxy_188 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 348, constant_args_idx = 499, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_979, 'MEAN': div_96, 'INVSTD': rsqrt_32, 'GAMMA': primals_196, 'BETA': primals_197, 'Y': permute_93, 'X_hat': permute_94, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_979 = div_96 = primals_197 = permute_93 = permute_94 = None
        getitem_414: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_188['Y']
        getitem_415: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_188['X_hat'];  triton_kernel_wrapper_functional_proxy_188 = None
        empty_290: "i32[100352, 16]" = torch.ops.aten.empty.memory_format([100352, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_291: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_292: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_985: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_415, [512, -1, 256]);  getitem_415 = None
        view_986: "bf16[100352, 256]" = torch.ops.aten.view.default(view_985, [100352, 256]);  view_985 = None
        triton_kernel_wrapper_functional_proxy_189 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 349, constant_args_idx = 500, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_986, 'P_ptr': empty_290, 'S_ptr': empty_291, 'M_ptr': empty_292, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_986 = empty_290 = empty_291 = empty_292 = None
        getitem_416: "i32[100352, 16]" = triton_kernel_wrapper_functional_proxy_189['P_ptr']
        getitem_417: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_189['S_ptr']
        getitem_418: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_189['M_ptr'];  triton_kernel_wrapper_functional_proxy_189 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_992: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(getitem_414, [512, 256, 14, 14]);  getitem_414 = None
        empty_293: "bf16[512, 256, 14, 14]" = torch.ops.aten.empty.memory_format([512, 256, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_95: "bf16[512, 256, 14, 14]" = torch.ops.aten.permute.default(empty_293, [0, 1, 2, 3]);  empty_293 = None
        triton_kernel_wrapper_functional_proxy_190 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 501, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_992, 'Y_ptr': permute_95, 'Mask_prt': full_default_150, 'n_elts': 25690112, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_992 = permute_95 = None
        getitem_419: "bf16[512, 256, 14, 14]" = triton_kernel_wrapper_functional_proxy_190['Y_ptr']
        getitem_420: "i8[512, 256, 14, 14]" = triton_kernel_wrapper_functional_proxy_190['Mask_prt'];  triton_kernel_wrapper_functional_proxy_190 = None
        view_997: "i8[512, 196, 256]" = torch.ops.aten.view.default(getitem_420, [512, -1, 256]);  getitem_420 = None
        view_998: "i8[100352, 256]" = torch.ops.aten.view.default(view_997, [100352, 256]);  view_997 = None
        triton_kernel_wrapper_functional_proxy_191 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 502, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_998, 'P_ptr': full_default_151, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_998 = None
        getitem_421: "i32[100352, 8]" = triton_kernel_wrapper_functional_proxy_191['P_ptr'];  triton_kernel_wrapper_functional_proxy_191 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_34: "bf16[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_200, torch.bfloat16);  primals_200 = None
        empty_294: "i32[100352, 16]" = torch.ops.aten.empty.memory_format([100352, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_295: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_296: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1003: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_419, [512, -1, 256])
        view_1004: "bf16[100352, 256]" = torch.ops.aten.view.default(view_1003, [100352, 256]);  view_1003 = None
        triton_kernel_wrapper_functional_proxy_192 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 350, constant_args_idx = 503, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1004, 'P_ptr': empty_294, 'S_ptr': empty_295, 'M_ptr': empty_296, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1004 = empty_294 = empty_295 = empty_296 = None
        getitem_422: "i32[100352, 16]" = triton_kernel_wrapper_functional_proxy_192['P_ptr']
        getitem_423: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_192['S_ptr']
        getitem_424: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_192['M_ptr'];  triton_kernel_wrapper_functional_proxy_192 = None
        convolution_33: "bf16[512, 1024, 14, 14]" = torch.ops.aten.convolution.default(getitem_419, convert_element_type_34, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_419 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_141: "i64[]" = torch.ops.aten.add.Tensor(primals_201, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_1010: "bf16[512, 1024, 196]" = torch.ops.aten.view.default(convolution_33, [512, 1024, 196]);  convolution_33 = None
        triton_kernel_wrapper_functional_proxy_193 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 351, constant_args_idx = 504, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1010, 'SUM': full_default_152, 'SUMSQ': full_default_152, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_425: "f32[1024]" = triton_kernel_wrapper_functional_proxy_193['SUM']
        getitem_426: "f32[1024]" = triton_kernel_wrapper_functional_proxy_193['SUMSQ'];  triton_kernel_wrapper_functional_proxy_193 = None
        div_99: "f32[1024]" = torch.ops.aten.div.Tensor(getitem_425, full_default_148);  getitem_425 = None
        div_100: "f32[1024]" = torch.ops.aten.div.Tensor(getitem_426, full_default_148);  getitem_426 = None
        mul_232: "f32[1024]" = torch.ops.aten.mul.Tensor(div_99, div_99)
        sub_66: "f32[1024]" = torch.ops.aten.sub.Tensor(div_100, mul_232);  div_100 = mul_232 = None
        clamp_min_66: "f32[1024]" = torch.ops.aten.clamp_min.default(sub_66, 0.0);  sub_66 = None
        add_142: "f32[1024]" = torch.ops.aten.add.Tensor(clamp_min_66, 1e-05)
        rsqrt_33: "f32[1024]" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
        mul_233: "f32[1024]" = torch.ops.aten.mul.Tensor(clamp_min_66, full_default_149);  clamp_min_66 = None
        mul_234: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_204, 0.9)
        mul_235: "f32[1024]" = torch.ops.aten.mul.Tensor(div_99, 0.1)
        add_143: "f32[1024]" = torch.ops.aten.add.Tensor(mul_234, mul_235);  mul_234 = mul_235 = None
        mul_236: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_205, 0.9)
        mul_237: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_233, 0.1);  mul_233 = None
        add_144: "f32[1024]" = torch.ops.aten.add.Tensor(mul_236, mul_237);  mul_236 = mul_237 = None
        empty_297: "bf16[512, 1024, 196]" = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_96: "bf16[512, 1024, 196]" = torch.ops.aten.permute.default(empty_297, [0, 1, 2]);  empty_297 = None
        empty_298: "bf16[512, 1024, 196]" = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_97: "bf16[512, 1024, 196]" = torch.ops.aten.permute.default(empty_298, [0, 1, 2]);  empty_298 = None
        triton_kernel_wrapper_functional_proxy_194 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 352, constant_args_idx = 505, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1010, 'MEAN': div_99, 'INVSTD': rsqrt_33, 'GAMMA': primals_202, 'BETA': primals_203, 'Y': permute_96, 'X_hat': permute_97, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_1010 = div_99 = primals_203 = permute_96 = permute_97 = None
        getitem_427: "bf16[512, 1024, 196]" = triton_kernel_wrapper_functional_proxy_194['Y']
        getitem_428: "bf16[512, 1024, 196]" = triton_kernel_wrapper_functional_proxy_194['X_hat'];  triton_kernel_wrapper_functional_proxy_194 = None
        empty_299: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_300: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_301: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1016: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_428, [512, -1, 256]);  getitem_428 = None
        view_1017: "bf16[401408, 256]" = torch.ops.aten.view.default(view_1016, [401408, 256]);  view_1016 = None
        triton_kernel_wrapper_functional_proxy_195 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 353, constant_args_idx = 506, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1017, 'P_ptr': empty_299, 'S_ptr': empty_300, 'M_ptr': empty_301, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1017 = empty_299 = empty_300 = empty_301 = None
        getitem_429: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_195['P_ptr']
        getitem_430: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_195['S_ptr']
        getitem_431: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_195['M_ptr'];  triton_kernel_wrapper_functional_proxy_195 = None
        
         # File: <eval_with_key>.3 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:113 in forward, code: add_9 = layer3_2_bn3 + layer3_1_relu_2;  layer3_2_bn3 = layer3_1_relu_2 = None
        view_1023: "bf16[512, 1024, 14, 14]" = torch.ops.aten.view.default(getitem_427, [512, 1024, 14, 14]);  getitem_427 = None
        add_145: "bf16[512, 1024, 14, 14]" = torch.ops.aten.add.Tensor(view_1023, getitem_393);  view_1023 = getitem_393 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        empty_302: "bf16[512, 1024, 14, 14]" = torch.ops.aten.empty.memory_format([512, 1024, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_98: "bf16[512, 1024, 14, 14]" = torch.ops.aten.permute.default(empty_302, [0, 1, 2, 3]);  empty_302 = None
        triton_kernel_wrapper_functional_proxy_196 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 507, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_145, 'Y_ptr': permute_98, 'Mask_prt': full_default_160, 'n_elts': 102760448, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  add_145 = permute_98 = None
        getitem_432: "bf16[512, 1024, 14, 14]" = triton_kernel_wrapper_functional_proxy_196['Y_ptr']
        getitem_433: "i8[512, 1024, 14, 14]" = triton_kernel_wrapper_functional_proxy_196['Mask_prt'];  triton_kernel_wrapper_functional_proxy_196 = None
        view_1028: "i8[512, 784, 256]" = torch.ops.aten.view.default(getitem_433, [512, -1, 256]);  getitem_433 = None
        view_1029: "i8[401408, 256]" = torch.ops.aten.view.default(view_1028, [401408, 256]);  view_1028 = None
        triton_kernel_wrapper_functional_proxy_197 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 508, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1029, 'P_ptr': full_default_11, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_1029 = None
        getitem_434: "i32[401408, 8]" = triton_kernel_wrapper_functional_proxy_197['P_ptr'];  triton_kernel_wrapper_functional_proxy_197 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_35: "bf16[256, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_206, torch.bfloat16);  primals_206 = None
        empty_303: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_304: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_305: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1034: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_432, [512, -1, 256])
        view_1035: "bf16[401408, 256]" = torch.ops.aten.view.default(view_1034, [401408, 256]);  view_1034 = None
        triton_kernel_wrapper_functional_proxy_198 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 354, constant_args_idx = 509, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1035, 'P_ptr': empty_303, 'S_ptr': empty_304, 'M_ptr': empty_305, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1035 = empty_303 = empty_304 = empty_305 = None
        getitem_435: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_198['P_ptr']
        getitem_436: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_198['S_ptr']
        getitem_437: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_198['M_ptr'];  triton_kernel_wrapper_functional_proxy_198 = None
        convolution_34: "bf16[512, 256, 14, 14]" = torch.ops.aten.convolution.default(getitem_432, convert_element_type_35, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_146: "i64[]" = torch.ops.aten.add.Tensor(primals_207, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_1041: "bf16[512, 256, 196]" = torch.ops.aten.view.default(convolution_34, [512, 256, 196]);  convolution_34 = None
        triton_kernel_wrapper_functional_proxy_199 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 355, constant_args_idx = 510, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1041, 'SUM': full_default_18, 'SUMSQ': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_438: "f32[256]" = triton_kernel_wrapper_functional_proxy_199['SUM']
        getitem_439: "f32[256]" = triton_kernel_wrapper_functional_proxy_199['SUMSQ'];  triton_kernel_wrapper_functional_proxy_199 = None
        div_102: "f32[256]" = torch.ops.aten.div.Tensor(getitem_438, full_default_148);  getitem_438 = None
        div_103: "f32[256]" = torch.ops.aten.div.Tensor(getitem_439, full_default_148);  getitem_439 = None
        mul_239: "f32[256]" = torch.ops.aten.mul.Tensor(div_102, div_102)
        sub_68: "f32[256]" = torch.ops.aten.sub.Tensor(div_103, mul_239);  div_103 = mul_239 = None
        clamp_min_68: "f32[256]" = torch.ops.aten.clamp_min.default(sub_68, 0.0);  sub_68 = None
        add_147: "f32[256]" = torch.ops.aten.add.Tensor(clamp_min_68, 1e-05)
        rsqrt_34: "f32[256]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
        mul_240: "f32[256]" = torch.ops.aten.mul.Tensor(clamp_min_68, full_default_149);  clamp_min_68 = None
        mul_241: "f32[256]" = torch.ops.aten.mul.Tensor(primals_210, 0.9)
        mul_242: "f32[256]" = torch.ops.aten.mul.Tensor(div_102, 0.1)
        add_148: "f32[256]" = torch.ops.aten.add.Tensor(mul_241, mul_242);  mul_241 = mul_242 = None
        mul_243: "f32[256]" = torch.ops.aten.mul.Tensor(primals_211, 0.9)
        mul_244: "f32[256]" = torch.ops.aten.mul.Tensor(mul_240, 0.1);  mul_240 = None
        add_149: "f32[256]" = torch.ops.aten.add.Tensor(mul_243, mul_244);  mul_243 = mul_244 = None
        empty_306: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_99: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_306, [0, 1, 2]);  empty_306 = None
        empty_307: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_100: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_307, [0, 1, 2]);  empty_307 = None
        triton_kernel_wrapper_functional_proxy_200 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 356, constant_args_idx = 511, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1041, 'MEAN': div_102, 'INVSTD': rsqrt_34, 'GAMMA': primals_208, 'BETA': primals_209, 'Y': permute_99, 'X_hat': permute_100, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_1041 = div_102 = primals_209 = permute_99 = permute_100 = None
        getitem_440: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_200['Y']
        getitem_441: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_200['X_hat'];  triton_kernel_wrapper_functional_proxy_200 = None
        empty_308: "i32[100352, 16]" = torch.ops.aten.empty.memory_format([100352, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_309: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_310: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1047: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_441, [512, -1, 256]);  getitem_441 = None
        view_1048: "bf16[100352, 256]" = torch.ops.aten.view.default(view_1047, [100352, 256]);  view_1047 = None
        triton_kernel_wrapper_functional_proxy_201 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 357, constant_args_idx = 512, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1048, 'P_ptr': empty_308, 'S_ptr': empty_309, 'M_ptr': empty_310, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1048 = empty_308 = empty_309 = empty_310 = None
        getitem_442: "i32[100352, 16]" = triton_kernel_wrapper_functional_proxy_201['P_ptr']
        getitem_443: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_201['S_ptr']
        getitem_444: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_201['M_ptr'];  triton_kernel_wrapper_functional_proxy_201 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1054: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(getitem_440, [512, 256, 14, 14]);  getitem_440 = None
        empty_311: "bf16[512, 256, 14, 14]" = torch.ops.aten.empty.memory_format([512, 256, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_101: "bf16[512, 256, 14, 14]" = torch.ops.aten.permute.default(empty_311, [0, 1, 2, 3]);  empty_311 = None
        triton_kernel_wrapper_functional_proxy_202 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 513, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1054, 'Y_ptr': permute_101, 'Mask_prt': full_default_150, 'n_elts': 25690112, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_1054 = permute_101 = None
        getitem_445: "bf16[512, 256, 14, 14]" = triton_kernel_wrapper_functional_proxy_202['Y_ptr']
        getitem_446: "i8[512, 256, 14, 14]" = triton_kernel_wrapper_functional_proxy_202['Mask_prt'];  triton_kernel_wrapper_functional_proxy_202 = None
        view_1059: "i8[512, 196, 256]" = torch.ops.aten.view.default(getitem_446, [512, -1, 256]);  getitem_446 = None
        view_1060: "i8[100352, 256]" = torch.ops.aten.view.default(view_1059, [100352, 256]);  view_1059 = None
        triton_kernel_wrapper_functional_proxy_203 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 514, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1060, 'P_ptr': full_default_151, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_1060 = None
        getitem_447: "i32[100352, 8]" = triton_kernel_wrapper_functional_proxy_203['P_ptr'];  triton_kernel_wrapper_functional_proxy_203 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_36: "bf16[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_212, torch.bfloat16);  primals_212 = None
        empty_312: "i32[100352, 16]" = torch.ops.aten.empty.memory_format([100352, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_313: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_314: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1065: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_445, [512, -1, 256])
        view_1066: "bf16[100352, 256]" = torch.ops.aten.view.default(view_1065, [100352, 256]);  view_1065 = None
        triton_kernel_wrapper_functional_proxy_204 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 358, constant_args_idx = 515, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1066, 'P_ptr': empty_312, 'S_ptr': empty_313, 'M_ptr': empty_314, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1066 = empty_312 = empty_313 = empty_314 = None
        getitem_448: "i32[100352, 16]" = triton_kernel_wrapper_functional_proxy_204['P_ptr']
        getitem_449: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_204['S_ptr']
        getitem_450: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_204['M_ptr'];  triton_kernel_wrapper_functional_proxy_204 = None
        convolution_35: "bf16[512, 256, 14, 14]" = torch.ops.aten.convolution.default(getitem_445, convert_element_type_36, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_445 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_150: "i64[]" = torch.ops.aten.add.Tensor(primals_213, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_1072: "bf16[512, 256, 196]" = torch.ops.aten.view.default(convolution_35, [512, 256, 196]);  convolution_35 = None
        triton_kernel_wrapper_functional_proxy_205 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 359, constant_args_idx = 516, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1072, 'SUM': full_default_18, 'SUMSQ': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_451: "f32[256]" = triton_kernel_wrapper_functional_proxy_205['SUM']
        getitem_452: "f32[256]" = triton_kernel_wrapper_functional_proxy_205['SUMSQ'];  triton_kernel_wrapper_functional_proxy_205 = None
        div_105: "f32[256]" = torch.ops.aten.div.Tensor(getitem_451, full_default_148);  getitem_451 = None
        div_106: "f32[256]" = torch.ops.aten.div.Tensor(getitem_452, full_default_148);  getitem_452 = None
        mul_246: "f32[256]" = torch.ops.aten.mul.Tensor(div_105, div_105)
        sub_70: "f32[256]" = torch.ops.aten.sub.Tensor(div_106, mul_246);  div_106 = mul_246 = None
        clamp_min_70: "f32[256]" = torch.ops.aten.clamp_min.default(sub_70, 0.0);  sub_70 = None
        add_151: "f32[256]" = torch.ops.aten.add.Tensor(clamp_min_70, 1e-05)
        rsqrt_35: "f32[256]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
        mul_247: "f32[256]" = torch.ops.aten.mul.Tensor(clamp_min_70, full_default_149);  clamp_min_70 = None
        mul_248: "f32[256]" = torch.ops.aten.mul.Tensor(primals_216, 0.9)
        mul_249: "f32[256]" = torch.ops.aten.mul.Tensor(div_105, 0.1)
        add_152: "f32[256]" = torch.ops.aten.add.Tensor(mul_248, mul_249);  mul_248 = mul_249 = None
        mul_250: "f32[256]" = torch.ops.aten.mul.Tensor(primals_217, 0.9)
        mul_251: "f32[256]" = torch.ops.aten.mul.Tensor(mul_247, 0.1);  mul_247 = None
        add_153: "f32[256]" = torch.ops.aten.add.Tensor(mul_250, mul_251);  mul_250 = mul_251 = None
        empty_315: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_102: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_315, [0, 1, 2]);  empty_315 = None
        empty_316: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_103: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_316, [0, 1, 2]);  empty_316 = None
        triton_kernel_wrapper_functional_proxy_206 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 360, constant_args_idx = 517, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1072, 'MEAN': div_105, 'INVSTD': rsqrt_35, 'GAMMA': primals_214, 'BETA': primals_215, 'Y': permute_102, 'X_hat': permute_103, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_1072 = div_105 = primals_215 = permute_102 = permute_103 = None
        getitem_453: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_206['Y']
        getitem_454: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_206['X_hat'];  triton_kernel_wrapper_functional_proxy_206 = None
        empty_317: "i32[100352, 16]" = torch.ops.aten.empty.memory_format([100352, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_318: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_319: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1078: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_454, [512, -1, 256]);  getitem_454 = None
        view_1079: "bf16[100352, 256]" = torch.ops.aten.view.default(view_1078, [100352, 256]);  view_1078 = None
        triton_kernel_wrapper_functional_proxy_207 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 361, constant_args_idx = 518, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1079, 'P_ptr': empty_317, 'S_ptr': empty_318, 'M_ptr': empty_319, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1079 = empty_317 = empty_318 = empty_319 = None
        getitem_455: "i32[100352, 16]" = triton_kernel_wrapper_functional_proxy_207['P_ptr']
        getitem_456: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_207['S_ptr']
        getitem_457: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_207['M_ptr'];  triton_kernel_wrapper_functional_proxy_207 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1085: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(getitem_453, [512, 256, 14, 14]);  getitem_453 = None
        empty_320: "bf16[512, 256, 14, 14]" = torch.ops.aten.empty.memory_format([512, 256, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_104: "bf16[512, 256, 14, 14]" = torch.ops.aten.permute.default(empty_320, [0, 1, 2, 3]);  empty_320 = None
        triton_kernel_wrapper_functional_proxy_208 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 519, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1085, 'Y_ptr': permute_104, 'Mask_prt': full_default_150, 'n_elts': 25690112, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_1085 = permute_104 = None
        getitem_458: "bf16[512, 256, 14, 14]" = triton_kernel_wrapper_functional_proxy_208['Y_ptr']
        getitem_459: "i8[512, 256, 14, 14]" = triton_kernel_wrapper_functional_proxy_208['Mask_prt'];  triton_kernel_wrapper_functional_proxy_208 = None
        view_1090: "i8[512, 196, 256]" = torch.ops.aten.view.default(getitem_459, [512, -1, 256]);  getitem_459 = None
        view_1091: "i8[100352, 256]" = torch.ops.aten.view.default(view_1090, [100352, 256]);  view_1090 = None
        triton_kernel_wrapper_functional_proxy_209 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 520, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1091, 'P_ptr': full_default_151, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_1091 = None
        getitem_460: "i32[100352, 8]" = triton_kernel_wrapper_functional_proxy_209['P_ptr'];  triton_kernel_wrapper_functional_proxy_209 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_37: "bf16[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_218, torch.bfloat16);  primals_218 = None
        empty_321: "i32[100352, 16]" = torch.ops.aten.empty.memory_format([100352, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_322: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_323: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1096: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_458, [512, -1, 256])
        view_1097: "bf16[100352, 256]" = torch.ops.aten.view.default(view_1096, [100352, 256]);  view_1096 = None
        triton_kernel_wrapper_functional_proxy_210 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 362, constant_args_idx = 521, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1097, 'P_ptr': empty_321, 'S_ptr': empty_322, 'M_ptr': empty_323, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1097 = empty_321 = empty_322 = empty_323 = None
        getitem_461: "i32[100352, 16]" = triton_kernel_wrapper_functional_proxy_210['P_ptr']
        getitem_462: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_210['S_ptr']
        getitem_463: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_210['M_ptr'];  triton_kernel_wrapper_functional_proxy_210 = None
        convolution_36: "bf16[512, 1024, 14, 14]" = torch.ops.aten.convolution.default(getitem_458, convert_element_type_37, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_458 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_154: "i64[]" = torch.ops.aten.add.Tensor(primals_219, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_1103: "bf16[512, 1024, 196]" = torch.ops.aten.view.default(convolution_36, [512, 1024, 196]);  convolution_36 = None
        triton_kernel_wrapper_functional_proxy_211 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 363, constant_args_idx = 522, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1103, 'SUM': full_default_152, 'SUMSQ': full_default_152, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_464: "f32[1024]" = triton_kernel_wrapper_functional_proxy_211['SUM']
        getitem_465: "f32[1024]" = triton_kernel_wrapper_functional_proxy_211['SUMSQ'];  triton_kernel_wrapper_functional_proxy_211 = None
        div_108: "f32[1024]" = torch.ops.aten.div.Tensor(getitem_464, full_default_148);  getitem_464 = None
        div_109: "f32[1024]" = torch.ops.aten.div.Tensor(getitem_465, full_default_148);  getitem_465 = None
        mul_253: "f32[1024]" = torch.ops.aten.mul.Tensor(div_108, div_108)
        sub_72: "f32[1024]" = torch.ops.aten.sub.Tensor(div_109, mul_253);  div_109 = mul_253 = None
        clamp_min_72: "f32[1024]" = torch.ops.aten.clamp_min.default(sub_72, 0.0);  sub_72 = None
        add_155: "f32[1024]" = torch.ops.aten.add.Tensor(clamp_min_72, 1e-05)
        rsqrt_36: "f32[1024]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
        mul_254: "f32[1024]" = torch.ops.aten.mul.Tensor(clamp_min_72, full_default_149);  clamp_min_72 = None
        mul_255: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_222, 0.9)
        mul_256: "f32[1024]" = torch.ops.aten.mul.Tensor(div_108, 0.1)
        add_156: "f32[1024]" = torch.ops.aten.add.Tensor(mul_255, mul_256);  mul_255 = mul_256 = None
        mul_257: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_223, 0.9)
        mul_258: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_254, 0.1);  mul_254 = None
        add_157: "f32[1024]" = torch.ops.aten.add.Tensor(mul_257, mul_258);  mul_257 = mul_258 = None
        empty_324: "bf16[512, 1024, 196]" = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_105: "bf16[512, 1024, 196]" = torch.ops.aten.permute.default(empty_324, [0, 1, 2]);  empty_324 = None
        empty_325: "bf16[512, 1024, 196]" = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_106: "bf16[512, 1024, 196]" = torch.ops.aten.permute.default(empty_325, [0, 1, 2]);  empty_325 = None
        triton_kernel_wrapper_functional_proxy_212 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 364, constant_args_idx = 523, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1103, 'MEAN': div_108, 'INVSTD': rsqrt_36, 'GAMMA': primals_220, 'BETA': primals_221, 'Y': permute_105, 'X_hat': permute_106, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_1103 = div_108 = primals_221 = permute_105 = permute_106 = None
        getitem_466: "bf16[512, 1024, 196]" = triton_kernel_wrapper_functional_proxy_212['Y']
        getitem_467: "bf16[512, 1024, 196]" = triton_kernel_wrapper_functional_proxy_212['X_hat'];  triton_kernel_wrapper_functional_proxy_212 = None
        empty_326: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_327: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_328: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1109: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_467, [512, -1, 256]);  getitem_467 = None
        view_1110: "bf16[401408, 256]" = torch.ops.aten.view.default(view_1109, [401408, 256]);  view_1109 = None
        triton_kernel_wrapper_functional_proxy_213 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 365, constant_args_idx = 524, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1110, 'P_ptr': empty_326, 'S_ptr': empty_327, 'M_ptr': empty_328, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1110 = empty_326 = empty_327 = empty_328 = None
        getitem_468: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_213['P_ptr']
        getitem_469: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_213['S_ptr']
        getitem_470: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_213['M_ptr'];  triton_kernel_wrapper_functional_proxy_213 = None
        
         # File: <eval_with_key>.3 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:123 in forward, code: add_10 = layer3_3_bn3 + layer3_2_relu_2;  layer3_3_bn3 = layer3_2_relu_2 = None
        view_1116: "bf16[512, 1024, 14, 14]" = torch.ops.aten.view.default(getitem_466, [512, 1024, 14, 14]);  getitem_466 = None
        add_158: "bf16[512, 1024, 14, 14]" = torch.ops.aten.add.Tensor(view_1116, getitem_432);  view_1116 = getitem_432 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        empty_329: "bf16[512, 1024, 14, 14]" = torch.ops.aten.empty.memory_format([512, 1024, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_107: "bf16[512, 1024, 14, 14]" = torch.ops.aten.permute.default(empty_329, [0, 1, 2, 3]);  empty_329 = None
        triton_kernel_wrapper_functional_proxy_214 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 525, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_158, 'Y_ptr': permute_107, 'Mask_prt': full_default_160, 'n_elts': 102760448, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  add_158 = permute_107 = None
        getitem_471: "bf16[512, 1024, 14, 14]" = triton_kernel_wrapper_functional_proxy_214['Y_ptr']
        getitem_472: "i8[512, 1024, 14, 14]" = triton_kernel_wrapper_functional_proxy_214['Mask_prt'];  triton_kernel_wrapper_functional_proxy_214 = None
        view_1121: "i8[512, 784, 256]" = torch.ops.aten.view.default(getitem_472, [512, -1, 256]);  getitem_472 = None
        view_1122: "i8[401408, 256]" = torch.ops.aten.view.default(view_1121, [401408, 256]);  view_1121 = None
        triton_kernel_wrapper_functional_proxy_215 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 526, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1122, 'P_ptr': full_default_11, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_1122 = None
        getitem_473: "i32[401408, 8]" = triton_kernel_wrapper_functional_proxy_215['P_ptr'];  triton_kernel_wrapper_functional_proxy_215 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_38: "bf16[256, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_224, torch.bfloat16);  primals_224 = None
        empty_330: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_331: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_332: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1127: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_471, [512, -1, 256])
        view_1128: "bf16[401408, 256]" = torch.ops.aten.view.default(view_1127, [401408, 256]);  view_1127 = None
        triton_kernel_wrapper_functional_proxy_216 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 366, constant_args_idx = 527, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1128, 'P_ptr': empty_330, 'S_ptr': empty_331, 'M_ptr': empty_332, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1128 = empty_330 = empty_331 = empty_332 = None
        getitem_474: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_216['P_ptr']
        getitem_475: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_216['S_ptr']
        getitem_476: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_216['M_ptr'];  triton_kernel_wrapper_functional_proxy_216 = None
        convolution_37: "bf16[512, 256, 14, 14]" = torch.ops.aten.convolution.default(getitem_471, convert_element_type_38, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_159: "i64[]" = torch.ops.aten.add.Tensor(primals_225, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_1134: "bf16[512, 256, 196]" = torch.ops.aten.view.default(convolution_37, [512, 256, 196]);  convolution_37 = None
        triton_kernel_wrapper_functional_proxy_217 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 367, constant_args_idx = 528, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1134, 'SUM': full_default_18, 'SUMSQ': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_477: "f32[256]" = triton_kernel_wrapper_functional_proxy_217['SUM']
        getitem_478: "f32[256]" = triton_kernel_wrapper_functional_proxy_217['SUMSQ'];  triton_kernel_wrapper_functional_proxy_217 = None
        div_111: "f32[256]" = torch.ops.aten.div.Tensor(getitem_477, full_default_148);  getitem_477 = None
        div_112: "f32[256]" = torch.ops.aten.div.Tensor(getitem_478, full_default_148);  getitem_478 = None
        mul_260: "f32[256]" = torch.ops.aten.mul.Tensor(div_111, div_111)
        sub_74: "f32[256]" = torch.ops.aten.sub.Tensor(div_112, mul_260);  div_112 = mul_260 = None
        clamp_min_74: "f32[256]" = torch.ops.aten.clamp_min.default(sub_74, 0.0);  sub_74 = None
        add_160: "f32[256]" = torch.ops.aten.add.Tensor(clamp_min_74, 1e-05)
        rsqrt_37: "f32[256]" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
        mul_261: "f32[256]" = torch.ops.aten.mul.Tensor(clamp_min_74, full_default_149);  clamp_min_74 = None
        mul_262: "f32[256]" = torch.ops.aten.mul.Tensor(primals_228, 0.9)
        mul_263: "f32[256]" = torch.ops.aten.mul.Tensor(div_111, 0.1)
        add_161: "f32[256]" = torch.ops.aten.add.Tensor(mul_262, mul_263);  mul_262 = mul_263 = None
        mul_264: "f32[256]" = torch.ops.aten.mul.Tensor(primals_229, 0.9)
        mul_265: "f32[256]" = torch.ops.aten.mul.Tensor(mul_261, 0.1);  mul_261 = None
        add_162: "f32[256]" = torch.ops.aten.add.Tensor(mul_264, mul_265);  mul_264 = mul_265 = None
        empty_333: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_108: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_333, [0, 1, 2]);  empty_333 = None
        empty_334: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_109: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_334, [0, 1, 2]);  empty_334 = None
        triton_kernel_wrapper_functional_proxy_218 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 368, constant_args_idx = 529, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1134, 'MEAN': div_111, 'INVSTD': rsqrt_37, 'GAMMA': primals_226, 'BETA': primals_227, 'Y': permute_108, 'X_hat': permute_109, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_1134 = div_111 = primals_227 = permute_108 = permute_109 = None
        getitem_479: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_218['Y']
        getitem_480: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_218['X_hat'];  triton_kernel_wrapper_functional_proxy_218 = None
        empty_335: "i32[100352, 16]" = torch.ops.aten.empty.memory_format([100352, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_336: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_337: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1140: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_480, [512, -1, 256]);  getitem_480 = None
        view_1141: "bf16[100352, 256]" = torch.ops.aten.view.default(view_1140, [100352, 256]);  view_1140 = None
        triton_kernel_wrapper_functional_proxy_219 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 369, constant_args_idx = 530, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1141, 'P_ptr': empty_335, 'S_ptr': empty_336, 'M_ptr': empty_337, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1141 = empty_335 = empty_336 = empty_337 = None
        getitem_481: "i32[100352, 16]" = triton_kernel_wrapper_functional_proxy_219['P_ptr']
        getitem_482: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_219['S_ptr']
        getitem_483: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_219['M_ptr'];  triton_kernel_wrapper_functional_proxy_219 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1147: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(getitem_479, [512, 256, 14, 14]);  getitem_479 = None
        empty_338: "bf16[512, 256, 14, 14]" = torch.ops.aten.empty.memory_format([512, 256, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_110: "bf16[512, 256, 14, 14]" = torch.ops.aten.permute.default(empty_338, [0, 1, 2, 3]);  empty_338 = None
        triton_kernel_wrapper_functional_proxy_220 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 531, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1147, 'Y_ptr': permute_110, 'Mask_prt': full_default_150, 'n_elts': 25690112, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_1147 = permute_110 = None
        getitem_484: "bf16[512, 256, 14, 14]" = triton_kernel_wrapper_functional_proxy_220['Y_ptr']
        getitem_485: "i8[512, 256, 14, 14]" = triton_kernel_wrapper_functional_proxy_220['Mask_prt'];  triton_kernel_wrapper_functional_proxy_220 = None
        view_1152: "i8[512, 196, 256]" = torch.ops.aten.view.default(getitem_485, [512, -1, 256]);  getitem_485 = None
        view_1153: "i8[100352, 256]" = torch.ops.aten.view.default(view_1152, [100352, 256]);  view_1152 = None
        triton_kernel_wrapper_functional_proxy_221 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 532, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1153, 'P_ptr': full_default_151, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_1153 = None
        getitem_486: "i32[100352, 8]" = triton_kernel_wrapper_functional_proxy_221['P_ptr'];  triton_kernel_wrapper_functional_proxy_221 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_39: "bf16[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_230, torch.bfloat16);  primals_230 = None
        empty_339: "i32[100352, 16]" = torch.ops.aten.empty.memory_format([100352, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_340: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_341: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1158: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_484, [512, -1, 256])
        view_1159: "bf16[100352, 256]" = torch.ops.aten.view.default(view_1158, [100352, 256]);  view_1158 = None
        triton_kernel_wrapper_functional_proxy_222 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 370, constant_args_idx = 533, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1159, 'P_ptr': empty_339, 'S_ptr': empty_340, 'M_ptr': empty_341, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1159 = empty_339 = empty_340 = empty_341 = None
        getitem_487: "i32[100352, 16]" = triton_kernel_wrapper_functional_proxy_222['P_ptr']
        getitem_488: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_222['S_ptr']
        getitem_489: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_222['M_ptr'];  triton_kernel_wrapper_functional_proxy_222 = None
        convolution_38: "bf16[512, 256, 14, 14]" = torch.ops.aten.convolution.default(getitem_484, convert_element_type_39, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_484 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_163: "i64[]" = torch.ops.aten.add.Tensor(primals_231, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_1165: "bf16[512, 256, 196]" = torch.ops.aten.view.default(convolution_38, [512, 256, 196]);  convolution_38 = None
        triton_kernel_wrapper_functional_proxy_223 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 371, constant_args_idx = 534, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1165, 'SUM': full_default_18, 'SUMSQ': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_490: "f32[256]" = triton_kernel_wrapper_functional_proxy_223['SUM']
        getitem_491: "f32[256]" = triton_kernel_wrapper_functional_proxy_223['SUMSQ'];  triton_kernel_wrapper_functional_proxy_223 = None
        div_114: "f32[256]" = torch.ops.aten.div.Tensor(getitem_490, full_default_148);  getitem_490 = None
        div_115: "f32[256]" = torch.ops.aten.div.Tensor(getitem_491, full_default_148);  getitem_491 = None
        mul_267: "f32[256]" = torch.ops.aten.mul.Tensor(div_114, div_114)
        sub_76: "f32[256]" = torch.ops.aten.sub.Tensor(div_115, mul_267);  div_115 = mul_267 = None
        clamp_min_76: "f32[256]" = torch.ops.aten.clamp_min.default(sub_76, 0.0);  sub_76 = None
        add_164: "f32[256]" = torch.ops.aten.add.Tensor(clamp_min_76, 1e-05)
        rsqrt_38: "f32[256]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
        mul_268: "f32[256]" = torch.ops.aten.mul.Tensor(clamp_min_76, full_default_149);  clamp_min_76 = None
        mul_269: "f32[256]" = torch.ops.aten.mul.Tensor(primals_234, 0.9)
        mul_270: "f32[256]" = torch.ops.aten.mul.Tensor(div_114, 0.1)
        add_165: "f32[256]" = torch.ops.aten.add.Tensor(mul_269, mul_270);  mul_269 = mul_270 = None
        mul_271: "f32[256]" = torch.ops.aten.mul.Tensor(primals_235, 0.9)
        mul_272: "f32[256]" = torch.ops.aten.mul.Tensor(mul_268, 0.1);  mul_268 = None
        add_166: "f32[256]" = torch.ops.aten.add.Tensor(mul_271, mul_272);  mul_271 = mul_272 = None
        empty_342: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_111: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_342, [0, 1, 2]);  empty_342 = None
        empty_343: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_112: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_343, [0, 1, 2]);  empty_343 = None
        triton_kernel_wrapper_functional_proxy_224 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 372, constant_args_idx = 535, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1165, 'MEAN': div_114, 'INVSTD': rsqrt_38, 'GAMMA': primals_232, 'BETA': primals_233, 'Y': permute_111, 'X_hat': permute_112, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_1165 = div_114 = primals_233 = permute_111 = permute_112 = None
        getitem_492: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_224['Y']
        getitem_493: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_224['X_hat'];  triton_kernel_wrapper_functional_proxy_224 = None
        empty_344: "i32[100352, 16]" = torch.ops.aten.empty.memory_format([100352, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_345: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_346: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1171: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_493, [512, -1, 256]);  getitem_493 = None
        view_1172: "bf16[100352, 256]" = torch.ops.aten.view.default(view_1171, [100352, 256]);  view_1171 = None
        triton_kernel_wrapper_functional_proxy_225 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 373, constant_args_idx = 536, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1172, 'P_ptr': empty_344, 'S_ptr': empty_345, 'M_ptr': empty_346, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1172 = empty_344 = empty_345 = empty_346 = None
        getitem_494: "i32[100352, 16]" = triton_kernel_wrapper_functional_proxy_225['P_ptr']
        getitem_495: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_225['S_ptr']
        getitem_496: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_225['M_ptr'];  triton_kernel_wrapper_functional_proxy_225 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1178: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(getitem_492, [512, 256, 14, 14]);  getitem_492 = None
        empty_347: "bf16[512, 256, 14, 14]" = torch.ops.aten.empty.memory_format([512, 256, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_113: "bf16[512, 256, 14, 14]" = torch.ops.aten.permute.default(empty_347, [0, 1, 2, 3]);  empty_347 = None
        triton_kernel_wrapper_functional_proxy_226 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 537, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1178, 'Y_ptr': permute_113, 'Mask_prt': full_default_150, 'n_elts': 25690112, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_1178 = permute_113 = None
        getitem_497: "bf16[512, 256, 14, 14]" = triton_kernel_wrapper_functional_proxy_226['Y_ptr']
        getitem_498: "i8[512, 256, 14, 14]" = triton_kernel_wrapper_functional_proxy_226['Mask_prt'];  triton_kernel_wrapper_functional_proxy_226 = None
        view_1183: "i8[512, 196, 256]" = torch.ops.aten.view.default(getitem_498, [512, -1, 256]);  getitem_498 = None
        view_1184: "i8[100352, 256]" = torch.ops.aten.view.default(view_1183, [100352, 256]);  view_1183 = None
        triton_kernel_wrapper_functional_proxy_227 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 538, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1184, 'P_ptr': full_default_151, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_1184 = None
        getitem_499: "i32[100352, 8]" = triton_kernel_wrapper_functional_proxy_227['P_ptr'];  triton_kernel_wrapper_functional_proxy_227 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_40: "bf16[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_236, torch.bfloat16);  primals_236 = None
        empty_348: "i32[100352, 16]" = torch.ops.aten.empty.memory_format([100352, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_349: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_350: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1189: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_497, [512, -1, 256])
        view_1190: "bf16[100352, 256]" = torch.ops.aten.view.default(view_1189, [100352, 256]);  view_1189 = None
        triton_kernel_wrapper_functional_proxy_228 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 374, constant_args_idx = 539, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1190, 'P_ptr': empty_348, 'S_ptr': empty_349, 'M_ptr': empty_350, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1190 = empty_348 = empty_349 = empty_350 = None
        getitem_500: "i32[100352, 16]" = triton_kernel_wrapper_functional_proxy_228['P_ptr']
        getitem_501: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_228['S_ptr']
        getitem_502: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_228['M_ptr'];  triton_kernel_wrapper_functional_proxy_228 = None
        convolution_39: "bf16[512, 1024, 14, 14]" = torch.ops.aten.convolution.default(getitem_497, convert_element_type_40, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_497 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_167: "i64[]" = torch.ops.aten.add.Tensor(primals_237, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_1196: "bf16[512, 1024, 196]" = torch.ops.aten.view.default(convolution_39, [512, 1024, 196]);  convolution_39 = None
        triton_kernel_wrapper_functional_proxy_229 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 375, constant_args_idx = 540, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1196, 'SUM': full_default_152, 'SUMSQ': full_default_152, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_503: "f32[1024]" = triton_kernel_wrapper_functional_proxy_229['SUM']
        getitem_504: "f32[1024]" = triton_kernel_wrapper_functional_proxy_229['SUMSQ'];  triton_kernel_wrapper_functional_proxy_229 = None
        div_117: "f32[1024]" = torch.ops.aten.div.Tensor(getitem_503, full_default_148);  getitem_503 = None
        div_118: "f32[1024]" = torch.ops.aten.div.Tensor(getitem_504, full_default_148);  getitem_504 = None
        mul_274: "f32[1024]" = torch.ops.aten.mul.Tensor(div_117, div_117)
        sub_78: "f32[1024]" = torch.ops.aten.sub.Tensor(div_118, mul_274);  div_118 = mul_274 = None
        clamp_min_78: "f32[1024]" = torch.ops.aten.clamp_min.default(sub_78, 0.0);  sub_78 = None
        add_168: "f32[1024]" = torch.ops.aten.add.Tensor(clamp_min_78, 1e-05)
        rsqrt_39: "f32[1024]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
        mul_275: "f32[1024]" = torch.ops.aten.mul.Tensor(clamp_min_78, full_default_149);  clamp_min_78 = None
        mul_276: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_240, 0.9)
        mul_277: "f32[1024]" = torch.ops.aten.mul.Tensor(div_117, 0.1)
        add_169: "f32[1024]" = torch.ops.aten.add.Tensor(mul_276, mul_277);  mul_276 = mul_277 = None
        mul_278: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_241, 0.9)
        mul_279: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_275, 0.1);  mul_275 = None
        add_170: "f32[1024]" = torch.ops.aten.add.Tensor(mul_278, mul_279);  mul_278 = mul_279 = None
        empty_351: "bf16[512, 1024, 196]" = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_114: "bf16[512, 1024, 196]" = torch.ops.aten.permute.default(empty_351, [0, 1, 2]);  empty_351 = None
        empty_352: "bf16[512, 1024, 196]" = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_115: "bf16[512, 1024, 196]" = torch.ops.aten.permute.default(empty_352, [0, 1, 2]);  empty_352 = None
        triton_kernel_wrapper_functional_proxy_230 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 376, constant_args_idx = 541, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1196, 'MEAN': div_117, 'INVSTD': rsqrt_39, 'GAMMA': primals_238, 'BETA': primals_239, 'Y': permute_114, 'X_hat': permute_115, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_1196 = div_117 = primals_239 = permute_114 = permute_115 = None
        getitem_505: "bf16[512, 1024, 196]" = triton_kernel_wrapper_functional_proxy_230['Y']
        getitem_506: "bf16[512, 1024, 196]" = triton_kernel_wrapper_functional_proxy_230['X_hat'];  triton_kernel_wrapper_functional_proxy_230 = None
        empty_353: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_354: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_355: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1202: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_506, [512, -1, 256]);  getitem_506 = None
        view_1203: "bf16[401408, 256]" = torch.ops.aten.view.default(view_1202, [401408, 256]);  view_1202 = None
        triton_kernel_wrapper_functional_proxy_231 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 377, constant_args_idx = 542, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1203, 'P_ptr': empty_353, 'S_ptr': empty_354, 'M_ptr': empty_355, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1203 = empty_353 = empty_354 = empty_355 = None
        getitem_507: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_231['P_ptr']
        getitem_508: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_231['S_ptr']
        getitem_509: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_231['M_ptr'];  triton_kernel_wrapper_functional_proxy_231 = None
        
         # File: <eval_with_key>.3 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:133 in forward, code: add_11 = layer3_4_bn3 + layer3_3_relu_2;  layer3_4_bn3 = layer3_3_relu_2 = None
        view_1209: "bf16[512, 1024, 14, 14]" = torch.ops.aten.view.default(getitem_505, [512, 1024, 14, 14]);  getitem_505 = None
        add_171: "bf16[512, 1024, 14, 14]" = torch.ops.aten.add.Tensor(view_1209, getitem_471);  view_1209 = getitem_471 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        empty_356: "bf16[512, 1024, 14, 14]" = torch.ops.aten.empty.memory_format([512, 1024, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_116: "bf16[512, 1024, 14, 14]" = torch.ops.aten.permute.default(empty_356, [0, 1, 2, 3]);  empty_356 = None
        triton_kernel_wrapper_functional_proxy_232 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 543, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_171, 'Y_ptr': permute_116, 'Mask_prt': full_default_160, 'n_elts': 102760448, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  add_171 = permute_116 = None
        getitem_510: "bf16[512, 1024, 14, 14]" = triton_kernel_wrapper_functional_proxy_232['Y_ptr']
        getitem_511: "i8[512, 1024, 14, 14]" = triton_kernel_wrapper_functional_proxy_232['Mask_prt'];  triton_kernel_wrapper_functional_proxy_232 = None
        view_1214: "i8[512, 784, 256]" = torch.ops.aten.view.default(getitem_511, [512, -1, 256]);  getitem_511 = None
        view_1215: "i8[401408, 256]" = torch.ops.aten.view.default(view_1214, [401408, 256]);  view_1214 = None
        triton_kernel_wrapper_functional_proxy_233 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 544, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1215, 'P_ptr': full_default_11, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_1215 = None
        getitem_512: "i32[401408, 8]" = triton_kernel_wrapper_functional_proxy_233['P_ptr'];  triton_kernel_wrapper_functional_proxy_233 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_41: "bf16[256, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_242, torch.bfloat16);  primals_242 = None
        empty_357: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_358: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_359: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1220: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_510, [512, -1, 256])
        view_1221: "bf16[401408, 256]" = torch.ops.aten.view.default(view_1220, [401408, 256]);  view_1220 = None
        triton_kernel_wrapper_functional_proxy_234 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 378, constant_args_idx = 545, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1221, 'P_ptr': empty_357, 'S_ptr': empty_358, 'M_ptr': empty_359, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1221 = empty_357 = empty_358 = empty_359 = None
        getitem_513: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_234['P_ptr']
        getitem_514: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_234['S_ptr']
        getitem_515: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_234['M_ptr'];  triton_kernel_wrapper_functional_proxy_234 = None
        convolution_40: "bf16[512, 256, 14, 14]" = torch.ops.aten.convolution.default(getitem_510, convert_element_type_41, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_172: "i64[]" = torch.ops.aten.add.Tensor(primals_243, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_1227: "bf16[512, 256, 196]" = torch.ops.aten.view.default(convolution_40, [512, 256, 196]);  convolution_40 = None
        triton_kernel_wrapper_functional_proxy_235 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 379, constant_args_idx = 546, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1227, 'SUM': full_default_18, 'SUMSQ': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_516: "f32[256]" = triton_kernel_wrapper_functional_proxy_235['SUM']
        getitem_517: "f32[256]" = triton_kernel_wrapper_functional_proxy_235['SUMSQ'];  triton_kernel_wrapper_functional_proxy_235 = None
        div_120: "f32[256]" = torch.ops.aten.div.Tensor(getitem_516, full_default_148);  getitem_516 = None
        div_121: "f32[256]" = torch.ops.aten.div.Tensor(getitem_517, full_default_148);  getitem_517 = None
        mul_281: "f32[256]" = torch.ops.aten.mul.Tensor(div_120, div_120)
        sub_80: "f32[256]" = torch.ops.aten.sub.Tensor(div_121, mul_281);  div_121 = mul_281 = None
        clamp_min_80: "f32[256]" = torch.ops.aten.clamp_min.default(sub_80, 0.0);  sub_80 = None
        add_173: "f32[256]" = torch.ops.aten.add.Tensor(clamp_min_80, 1e-05)
        rsqrt_40: "f32[256]" = torch.ops.aten.rsqrt.default(add_173);  add_173 = None
        mul_282: "f32[256]" = torch.ops.aten.mul.Tensor(clamp_min_80, full_default_149);  clamp_min_80 = None
        mul_283: "f32[256]" = torch.ops.aten.mul.Tensor(primals_246, 0.9)
        mul_284: "f32[256]" = torch.ops.aten.mul.Tensor(div_120, 0.1)
        add_174: "f32[256]" = torch.ops.aten.add.Tensor(mul_283, mul_284);  mul_283 = mul_284 = None
        mul_285: "f32[256]" = torch.ops.aten.mul.Tensor(primals_247, 0.9)
        mul_286: "f32[256]" = torch.ops.aten.mul.Tensor(mul_282, 0.1);  mul_282 = None
        add_175: "f32[256]" = torch.ops.aten.add.Tensor(mul_285, mul_286);  mul_285 = mul_286 = None
        empty_360: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_117: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_360, [0, 1, 2]);  empty_360 = None
        empty_361: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_118: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_361, [0, 1, 2]);  empty_361 = None
        triton_kernel_wrapper_functional_proxy_236 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 380, constant_args_idx = 547, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1227, 'MEAN': div_120, 'INVSTD': rsqrt_40, 'GAMMA': primals_244, 'BETA': primals_245, 'Y': permute_117, 'X_hat': permute_118, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_1227 = div_120 = primals_245 = permute_117 = permute_118 = None
        getitem_518: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_236['Y']
        getitem_519: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_236['X_hat'];  triton_kernel_wrapper_functional_proxy_236 = None
        empty_362: "i32[100352, 16]" = torch.ops.aten.empty.memory_format([100352, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_363: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_364: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1233: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_519, [512, -1, 256]);  getitem_519 = None
        view_1234: "bf16[100352, 256]" = torch.ops.aten.view.default(view_1233, [100352, 256]);  view_1233 = None
        triton_kernel_wrapper_functional_proxy_237 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 381, constant_args_idx = 548, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1234, 'P_ptr': empty_362, 'S_ptr': empty_363, 'M_ptr': empty_364, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1234 = empty_362 = empty_363 = empty_364 = None
        getitem_520: "i32[100352, 16]" = triton_kernel_wrapper_functional_proxy_237['P_ptr']
        getitem_521: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_237['S_ptr']
        getitem_522: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_237['M_ptr'];  triton_kernel_wrapper_functional_proxy_237 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1240: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(getitem_518, [512, 256, 14, 14]);  getitem_518 = None
        empty_365: "bf16[512, 256, 14, 14]" = torch.ops.aten.empty.memory_format([512, 256, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_119: "bf16[512, 256, 14, 14]" = torch.ops.aten.permute.default(empty_365, [0, 1, 2, 3]);  empty_365 = None
        triton_kernel_wrapper_functional_proxy_238 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 549, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1240, 'Y_ptr': permute_119, 'Mask_prt': full_default_150, 'n_elts': 25690112, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_1240 = permute_119 = None
        getitem_523: "bf16[512, 256, 14, 14]" = triton_kernel_wrapper_functional_proxy_238['Y_ptr']
        getitem_524: "i8[512, 256, 14, 14]" = triton_kernel_wrapper_functional_proxy_238['Mask_prt'];  triton_kernel_wrapper_functional_proxy_238 = None
        view_1245: "i8[512, 196, 256]" = torch.ops.aten.view.default(getitem_524, [512, -1, 256]);  getitem_524 = None
        view_1246: "i8[100352, 256]" = torch.ops.aten.view.default(view_1245, [100352, 256]);  view_1245 = None
        triton_kernel_wrapper_functional_proxy_239 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 550, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1246, 'P_ptr': full_default_151, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_1246 = None
        getitem_525: "i32[100352, 8]" = triton_kernel_wrapper_functional_proxy_239['P_ptr'];  triton_kernel_wrapper_functional_proxy_239 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_42: "bf16[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_248, torch.bfloat16);  primals_248 = None
        empty_366: "i32[100352, 16]" = torch.ops.aten.empty.memory_format([100352, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_367: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_368: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1251: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_523, [512, -1, 256])
        view_1252: "bf16[100352, 256]" = torch.ops.aten.view.default(view_1251, [100352, 256]);  view_1251 = None
        triton_kernel_wrapper_functional_proxy_240 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 382, constant_args_idx = 551, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1252, 'P_ptr': empty_366, 'S_ptr': empty_367, 'M_ptr': empty_368, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1252 = empty_366 = empty_367 = empty_368 = None
        getitem_526: "i32[100352, 16]" = triton_kernel_wrapper_functional_proxy_240['P_ptr']
        getitem_527: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_240['S_ptr']
        getitem_528: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_240['M_ptr'];  triton_kernel_wrapper_functional_proxy_240 = None
        convolution_41: "bf16[512, 256, 14, 14]" = torch.ops.aten.convolution.default(getitem_523, convert_element_type_42, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_523 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_176: "i64[]" = torch.ops.aten.add.Tensor(primals_249, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_1258: "bf16[512, 256, 196]" = torch.ops.aten.view.default(convolution_41, [512, 256, 196]);  convolution_41 = None
        triton_kernel_wrapper_functional_proxy_241 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 383, constant_args_idx = 552, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1258, 'SUM': full_default_18, 'SUMSQ': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ']);  full_default_18 = None
        getitem_529: "f32[256]" = triton_kernel_wrapper_functional_proxy_241['SUM']
        getitem_530: "f32[256]" = triton_kernel_wrapper_functional_proxy_241['SUMSQ'];  triton_kernel_wrapper_functional_proxy_241 = None
        div_123: "f32[256]" = torch.ops.aten.div.Tensor(getitem_529, full_default_148);  getitem_529 = None
        div_124: "f32[256]" = torch.ops.aten.div.Tensor(getitem_530, full_default_148);  getitem_530 = None
        mul_288: "f32[256]" = torch.ops.aten.mul.Tensor(div_123, div_123)
        sub_82: "f32[256]" = torch.ops.aten.sub.Tensor(div_124, mul_288);  div_124 = mul_288 = None
        clamp_min_82: "f32[256]" = torch.ops.aten.clamp_min.default(sub_82, 0.0);  sub_82 = None
        add_177: "f32[256]" = torch.ops.aten.add.Tensor(clamp_min_82, 1e-05)
        rsqrt_41: "f32[256]" = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
        mul_289: "f32[256]" = torch.ops.aten.mul.Tensor(clamp_min_82, full_default_149);  clamp_min_82 = None
        mul_290: "f32[256]" = torch.ops.aten.mul.Tensor(primals_252, 0.9)
        mul_291: "f32[256]" = torch.ops.aten.mul.Tensor(div_123, 0.1)
        add_178: "f32[256]" = torch.ops.aten.add.Tensor(mul_290, mul_291);  mul_290 = mul_291 = None
        mul_292: "f32[256]" = torch.ops.aten.mul.Tensor(primals_253, 0.9)
        mul_293: "f32[256]" = torch.ops.aten.mul.Tensor(mul_289, 0.1);  mul_289 = None
        add_179: "f32[256]" = torch.ops.aten.add.Tensor(mul_292, mul_293);  mul_292 = mul_293 = None
        empty_369: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_120: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_369, [0, 1, 2]);  empty_369 = None
        empty_370: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_121: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_370, [0, 1, 2]);  empty_370 = None
        triton_kernel_wrapper_functional_proxy_242 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 384, constant_args_idx = 553, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1258, 'MEAN': div_123, 'INVSTD': rsqrt_41, 'GAMMA': primals_250, 'BETA': primals_251, 'Y': permute_120, 'X_hat': permute_121, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_1258 = div_123 = primals_251 = permute_120 = permute_121 = None
        getitem_531: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_242['Y']
        getitem_532: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_242['X_hat'];  triton_kernel_wrapper_functional_proxy_242 = None
        empty_371: "i32[100352, 16]" = torch.ops.aten.empty.memory_format([100352, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_372: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_373: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1264: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_532, [512, -1, 256]);  getitem_532 = None
        view_1265: "bf16[100352, 256]" = torch.ops.aten.view.default(view_1264, [100352, 256]);  view_1264 = None
        triton_kernel_wrapper_functional_proxy_243 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 385, constant_args_idx = 554, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1265, 'P_ptr': empty_371, 'S_ptr': empty_372, 'M_ptr': empty_373, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1265 = empty_371 = empty_372 = empty_373 = None
        getitem_533: "i32[100352, 16]" = triton_kernel_wrapper_functional_proxy_243['P_ptr']
        getitem_534: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_243['S_ptr']
        getitem_535: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_243['M_ptr'];  triton_kernel_wrapper_functional_proxy_243 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1271: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(getitem_531, [512, 256, 14, 14]);  getitem_531 = None
        empty_374: "bf16[512, 256, 14, 14]" = torch.ops.aten.empty.memory_format([512, 256, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_122: "bf16[512, 256, 14, 14]" = torch.ops.aten.permute.default(empty_374, [0, 1, 2, 3]);  empty_374 = None
        triton_kernel_wrapper_functional_proxy_244 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 555, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1271, 'Y_ptr': permute_122, 'Mask_prt': full_default_150, 'n_elts': 25690112, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_1271 = permute_122 = full_default_150 = None
        getitem_536: "bf16[512, 256, 14, 14]" = triton_kernel_wrapper_functional_proxy_244['Y_ptr']
        getitem_537: "i8[512, 256, 14, 14]" = triton_kernel_wrapper_functional_proxy_244['Mask_prt'];  triton_kernel_wrapper_functional_proxy_244 = None
        view_1276: "i8[512, 196, 256]" = torch.ops.aten.view.default(getitem_537, [512, -1, 256]);  getitem_537 = None
        view_1277: "i8[100352, 256]" = torch.ops.aten.view.default(view_1276, [100352, 256]);  view_1276 = None
        triton_kernel_wrapper_functional_proxy_245 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 556, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1277, 'P_ptr': full_default_151, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_1277 = full_default_151 = None
        getitem_538: "i32[100352, 8]" = triton_kernel_wrapper_functional_proxy_245['P_ptr'];  triton_kernel_wrapper_functional_proxy_245 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_43: "bf16[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_254, torch.bfloat16);  primals_254 = None
        empty_375: "i32[100352, 16]" = torch.ops.aten.empty.memory_format([100352, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_376: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_377: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1282: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_536, [512, -1, 256])
        view_1283: "bf16[100352, 256]" = torch.ops.aten.view.default(view_1282, [100352, 256]);  view_1282 = None
        triton_kernel_wrapper_functional_proxy_246 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 386, constant_args_idx = 557, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1283, 'P_ptr': empty_375, 'S_ptr': empty_376, 'M_ptr': empty_377, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1283 = empty_375 = empty_376 = empty_377 = None
        getitem_539: "i32[100352, 16]" = triton_kernel_wrapper_functional_proxy_246['P_ptr']
        getitem_540: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_246['S_ptr']
        getitem_541: "bf16[100352]" = triton_kernel_wrapper_functional_proxy_246['M_ptr'];  triton_kernel_wrapper_functional_proxy_246 = None
        convolution_42: "bf16[512, 1024, 14, 14]" = torch.ops.aten.convolution.default(getitem_536, convert_element_type_43, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_536 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_180: "i64[]" = torch.ops.aten.add.Tensor(primals_255, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_1289: "bf16[512, 1024, 196]" = torch.ops.aten.view.default(convolution_42, [512, 1024, 196]);  convolution_42 = None
        triton_kernel_wrapper_functional_proxy_247 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 387, constant_args_idx = 558, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1289, 'SUM': full_default_152, 'SUMSQ': full_default_152, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ']);  full_default_152 = None
        getitem_542: "f32[1024]" = triton_kernel_wrapper_functional_proxy_247['SUM']
        getitem_543: "f32[1024]" = triton_kernel_wrapper_functional_proxy_247['SUMSQ'];  triton_kernel_wrapper_functional_proxy_247 = None
        div_126: "f32[1024]" = torch.ops.aten.div.Tensor(getitem_542, full_default_148);  getitem_542 = None
        div_127: "f32[1024]" = torch.ops.aten.div.Tensor(getitem_543, full_default_148);  getitem_543 = None
        mul_295: "f32[1024]" = torch.ops.aten.mul.Tensor(div_126, div_126)
        sub_84: "f32[1024]" = torch.ops.aten.sub.Tensor(div_127, mul_295);  div_127 = mul_295 = None
        clamp_min_84: "f32[1024]" = torch.ops.aten.clamp_min.default(sub_84, 0.0);  sub_84 = None
        add_181: "f32[1024]" = torch.ops.aten.add.Tensor(clamp_min_84, 1e-05)
        rsqrt_42: "f32[1024]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        mul_296: "f32[1024]" = torch.ops.aten.mul.Tensor(clamp_min_84, full_default_149);  clamp_min_84 = None
        mul_297: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_258, 0.9)
        mul_298: "f32[1024]" = torch.ops.aten.mul.Tensor(div_126, 0.1)
        add_182: "f32[1024]" = torch.ops.aten.add.Tensor(mul_297, mul_298);  mul_297 = mul_298 = None
        mul_299: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_259, 0.9)
        mul_300: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_296, 0.1);  mul_296 = None
        add_183: "f32[1024]" = torch.ops.aten.add.Tensor(mul_299, mul_300);  mul_299 = mul_300 = None
        empty_378: "bf16[512, 1024, 196]" = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_123: "bf16[512, 1024, 196]" = torch.ops.aten.permute.default(empty_378, [0, 1, 2]);  empty_378 = None
        empty_379: "bf16[512, 1024, 196]" = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_124: "bf16[512, 1024, 196]" = torch.ops.aten.permute.default(empty_379, [0, 1, 2]);  empty_379 = None
        triton_kernel_wrapper_functional_proxy_248 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 388, constant_args_idx = 559, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1289, 'MEAN': div_126, 'INVSTD': rsqrt_42, 'GAMMA': primals_256, 'BETA': primals_257, 'Y': permute_123, 'X_hat': permute_124, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_1289 = div_126 = primals_257 = permute_123 = permute_124 = None
        getitem_544: "bf16[512, 1024, 196]" = triton_kernel_wrapper_functional_proxy_248['Y']
        getitem_545: "bf16[512, 1024, 196]" = triton_kernel_wrapper_functional_proxy_248['X_hat'];  triton_kernel_wrapper_functional_proxy_248 = None
        empty_380: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_381: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_382: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1295: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_545, [512, -1, 256]);  getitem_545 = None
        view_1296: "bf16[401408, 256]" = torch.ops.aten.view.default(view_1295, [401408, 256]);  view_1295 = None
        triton_kernel_wrapper_functional_proxy_249 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 389, constant_args_idx = 560, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1296, 'P_ptr': empty_380, 'S_ptr': empty_381, 'M_ptr': empty_382, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1296 = empty_380 = empty_381 = empty_382 = None
        getitem_546: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_249['P_ptr']
        getitem_547: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_249['S_ptr']
        getitem_548: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_249['M_ptr'];  triton_kernel_wrapper_functional_proxy_249 = None
        
         # File: <eval_with_key>.3 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:143 in forward, code: add_12 = layer3_5_bn3 + layer3_4_relu_2;  layer3_5_bn3 = layer3_4_relu_2 = None
        view_1302: "bf16[512, 1024, 14, 14]" = torch.ops.aten.view.default(getitem_544, [512, 1024, 14, 14]);  getitem_544 = None
        add_184: "bf16[512, 1024, 14, 14]" = torch.ops.aten.add.Tensor(view_1302, getitem_510);  view_1302 = getitem_510 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        empty_383: "bf16[512, 1024, 14, 14]" = torch.ops.aten.empty.memory_format([512, 1024, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_125: "bf16[512, 1024, 14, 14]" = torch.ops.aten.permute.default(empty_383, [0, 1, 2, 3]);  empty_383 = None
        triton_kernel_wrapper_functional_proxy_250 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 561, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_184, 'Y_ptr': permute_125, 'Mask_prt': full_default_160, 'n_elts': 102760448, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  add_184 = permute_125 = full_default_160 = None
        getitem_549: "bf16[512, 1024, 14, 14]" = triton_kernel_wrapper_functional_proxy_250['Y_ptr']
        getitem_550: "i8[512, 1024, 14, 14]" = triton_kernel_wrapper_functional_proxy_250['Mask_prt'];  triton_kernel_wrapper_functional_proxy_250 = None
        view_1307: "i8[512, 784, 256]" = torch.ops.aten.view.default(getitem_550, [512, -1, 256]);  getitem_550 = None
        view_1308: "i8[401408, 256]" = torch.ops.aten.view.default(view_1307, [401408, 256]);  view_1307 = None
        triton_kernel_wrapper_functional_proxy_251 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 562, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1308, 'P_ptr': full_default_11, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_1308 = full_default_11 = None
        getitem_551: "i32[401408, 8]" = triton_kernel_wrapper_functional_proxy_251['P_ptr'];  triton_kernel_wrapper_functional_proxy_251 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_44: "bf16[512, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_260, torch.bfloat16);  primals_260 = None
        empty_384: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_385: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_386: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1313: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_549, [512, -1, 256])
        view_1314: "bf16[401408, 256]" = torch.ops.aten.view.default(view_1313, [401408, 256]);  view_1313 = None
        triton_kernel_wrapper_functional_proxy_252 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 390, constant_args_idx = 563, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1314, 'P_ptr': empty_384, 'S_ptr': empty_385, 'M_ptr': empty_386, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  empty_384 = empty_385 = empty_386 = None
        getitem_552: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_252['P_ptr']
        getitem_553: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_252['S_ptr']
        getitem_554: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_252['M_ptr'];  triton_kernel_wrapper_functional_proxy_252 = None
        convolution_43: "bf16[512, 512, 14, 14]" = torch.ops.aten.convolution.default(getitem_549, convert_element_type_44, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_185: "i64[]" = torch.ops.aten.add.Tensor(primals_261, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_1320: "bf16[512, 512, 196]" = torch.ops.aten.view.default(convolution_43, [512, 512, 196]);  convolution_43 = None
        triton_kernel_wrapper_functional_proxy_253 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 391, constant_args_idx = 564, grid = [(512, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1320, 'SUM': full_default_76, 'SUMSQ': full_default_76, 'M': 100352, 'HW': 196, 'stride_n': 100352, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_555: "f32[512]" = triton_kernel_wrapper_functional_proxy_253['SUM']
        getitem_556: "f32[512]" = triton_kernel_wrapper_functional_proxy_253['SUMSQ'];  triton_kernel_wrapper_functional_proxy_253 = None
        div_129: "f32[512]" = torch.ops.aten.div.Tensor(getitem_555, full_default_148);  getitem_555 = None
        div_130: "f32[512]" = torch.ops.aten.div.Tensor(getitem_556, full_default_148);  getitem_556 = full_default_148 = None
        mul_302: "f32[512]" = torch.ops.aten.mul.Tensor(div_129, div_129)
        sub_86: "f32[512]" = torch.ops.aten.sub.Tensor(div_130, mul_302);  div_130 = mul_302 = None
        clamp_min_86: "f32[512]" = torch.ops.aten.clamp_min.default(sub_86, 0.0);  sub_86 = None
        add_186: "f32[512]" = torch.ops.aten.add.Tensor(clamp_min_86, 1e-05)
        rsqrt_43: "f32[512]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
        mul_303: "f32[512]" = torch.ops.aten.mul.Tensor(clamp_min_86, full_default_149);  clamp_min_86 = full_default_149 = None
        mul_304: "f32[512]" = torch.ops.aten.mul.Tensor(primals_264, 0.9)
        mul_305: "f32[512]" = torch.ops.aten.mul.Tensor(div_129, 0.1)
        add_187: "f32[512]" = torch.ops.aten.add.Tensor(mul_304, mul_305);  mul_304 = mul_305 = None
        mul_306: "f32[512]" = torch.ops.aten.mul.Tensor(primals_265, 0.9)
        mul_307: "f32[512]" = torch.ops.aten.mul.Tensor(mul_303, 0.1);  mul_303 = None
        add_188: "f32[512]" = torch.ops.aten.add.Tensor(mul_306, mul_307);  mul_306 = mul_307 = None
        empty_387: "bf16[512, 512, 196]" = torch.ops.aten.empty.memory_format([512, 512, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_126: "bf16[512, 512, 196]" = torch.ops.aten.permute.default(empty_387, [0, 1, 2]);  empty_387 = None
        empty_388: "bf16[512, 512, 196]" = torch.ops.aten.empty.memory_format([512, 512, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_127: "bf16[512, 512, 196]" = torch.ops.aten.permute.default(empty_388, [0, 1, 2]);  empty_388 = None
        triton_kernel_wrapper_functional_proxy_254 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 392, constant_args_idx = 565, grid = [(512, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1320, 'MEAN': div_129, 'INVSTD': rsqrt_43, 'GAMMA': primals_262, 'BETA': primals_263, 'Y': permute_126, 'X_hat': permute_127, 'M': 100352, 'HW': 196, 'stride_n': 100352, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_1320 = div_129 = primals_263 = permute_126 = permute_127 = None
        getitem_557: "bf16[512, 512, 196]" = triton_kernel_wrapper_functional_proxy_254['Y']
        getitem_558: "bf16[512, 512, 196]" = triton_kernel_wrapper_functional_proxy_254['X_hat'];  triton_kernel_wrapper_functional_proxy_254 = None
        empty_389: "i32[200704, 16]" = torch.ops.aten.empty.memory_format([200704, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_390: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_391: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1326: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_558, [512, -1, 256]);  getitem_558 = None
        view_1327: "bf16[200704, 256]" = torch.ops.aten.view.default(view_1326, [200704, 256]);  view_1326 = None
        triton_kernel_wrapper_functional_proxy_255 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 393, constant_args_idx = 566, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1327, 'P_ptr': empty_389, 'S_ptr': empty_390, 'M_ptr': empty_391, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1327 = empty_389 = empty_390 = empty_391 = None
        getitem_559: "i32[200704, 16]" = triton_kernel_wrapper_functional_proxy_255['P_ptr']
        getitem_560: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_255['S_ptr']
        getitem_561: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_255['M_ptr'];  triton_kernel_wrapper_functional_proxy_255 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_256: "i8[512, 512, 14, 14]" = torch.ops.aten.full.default([512, 512, 14, 14], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_1333: "bf16[512, 512, 14, 14]" = torch.ops.aten.view.default(getitem_557, [512, 512, 14, 14]);  getitem_557 = None
        empty_392: "bf16[512, 512, 14, 14]" = torch.ops.aten.empty.memory_format([512, 512, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_128: "bf16[512, 512, 14, 14]" = torch.ops.aten.permute.default(empty_392, [0, 1, 2, 3]);  empty_392 = None
        triton_kernel_wrapper_functional_proxy_256 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 567, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1333, 'Y_ptr': permute_128, 'Mask_prt': full_default_256, 'n_elts': 51380224, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_1333 = permute_128 = full_default_256 = None
        getitem_562: "bf16[512, 512, 14, 14]" = triton_kernel_wrapper_functional_proxy_256['Y_ptr']
        getitem_563: "i8[512, 512, 14, 14]" = triton_kernel_wrapper_functional_proxy_256['Mask_prt'];  triton_kernel_wrapper_functional_proxy_256 = None
        view_1338: "i8[512, 392, 256]" = torch.ops.aten.view.default(getitem_563, [512, -1, 256]);  getitem_563 = None
        view_1339: "i8[200704, 256]" = torch.ops.aten.view.default(view_1338, [200704, 256]);  view_1338 = None
        triton_kernel_wrapper_functional_proxy_257 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 568, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1339, 'P_ptr': full_default_75, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_1339 = None
        getitem_564: "i32[200704, 8]" = triton_kernel_wrapper_functional_proxy_257['P_ptr'];  triton_kernel_wrapper_functional_proxy_257 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_45: "bf16[512, 512, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_266, torch.bfloat16);  primals_266 = None
        empty_393: "i32[200704, 16]" = torch.ops.aten.empty.memory_format([200704, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_394: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_395: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1344: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_562, [512, -1, 256])
        view_1345: "bf16[200704, 256]" = torch.ops.aten.view.default(view_1344, [200704, 256]);  view_1344 = None
        triton_kernel_wrapper_functional_proxy_258 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 394, constant_args_idx = 569, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1345, 'P_ptr': empty_393, 'S_ptr': empty_394, 'M_ptr': empty_395, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1345 = empty_393 = empty_394 = empty_395 = None
        getitem_565: "i32[200704, 16]" = triton_kernel_wrapper_functional_proxy_258['P_ptr']
        getitem_566: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_258['S_ptr']
        getitem_567: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_258['M_ptr'];  triton_kernel_wrapper_functional_proxy_258 = None
        convolution_44: "bf16[512, 512, 7, 7]" = torch.ops.aten.convolution.default(getitem_562, convert_element_type_45, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  getitem_562 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_189: "i64[]" = torch.ops.aten.add.Tensor(primals_267, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_1351: "bf16[512, 512, 49]" = torch.ops.aten.view.default(convolution_44, [512, 512, 49]);  convolution_44 = None
        triton_kernel_wrapper_functional_proxy_259 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 395, constant_args_idx = 570, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1351, 'SUM': full_default_76, 'SUMSQ': full_default_76, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_568: "f32[512]" = triton_kernel_wrapper_functional_proxy_259['SUM']
        getitem_569: "f32[512]" = triton_kernel_wrapper_functional_proxy_259['SUMSQ'];  triton_kernel_wrapper_functional_proxy_259 = None
        full_default_260: "f32[]" = torch.ops.aten.full.default([], 25088.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        div_132: "f32[512]" = torch.ops.aten.div.Tensor(getitem_568, full_default_260);  getitem_568 = None
        div_133: "f32[512]" = torch.ops.aten.div.Tensor(getitem_569, full_default_260);  getitem_569 = None
        mul_309: "f32[512]" = torch.ops.aten.mul.Tensor(div_132, div_132)
        sub_88: "f32[512]" = torch.ops.aten.sub.Tensor(div_133, mul_309);  div_133 = mul_309 = None
        clamp_min_88: "f32[512]" = torch.ops.aten.clamp_min.default(sub_88, 0.0);  sub_88 = None
        add_190: "f32[512]" = torch.ops.aten.add.Tensor(clamp_min_88, 1e-05)
        rsqrt_44: "f32[512]" = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
        full_default_261: "f32[]" = torch.ops.aten.full.default([], 1.00003981590271, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        mul_310: "f32[512]" = torch.ops.aten.mul.Tensor(clamp_min_88, full_default_261);  clamp_min_88 = None
        mul_311: "f32[512]" = torch.ops.aten.mul.Tensor(primals_270, 0.9)
        mul_312: "f32[512]" = torch.ops.aten.mul.Tensor(div_132, 0.1)
        add_191: "f32[512]" = torch.ops.aten.add.Tensor(mul_311, mul_312);  mul_311 = mul_312 = None
        mul_313: "f32[512]" = torch.ops.aten.mul.Tensor(primals_271, 0.9)
        mul_314: "f32[512]" = torch.ops.aten.mul.Tensor(mul_310, 0.1);  mul_310 = None
        add_192: "f32[512]" = torch.ops.aten.add.Tensor(mul_313, mul_314);  mul_313 = mul_314 = None
        empty_396: "bf16[512, 512, 49]" = torch.ops.aten.empty.memory_format([512, 512, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_129: "bf16[512, 512, 49]" = torch.ops.aten.permute.default(empty_396, [0, 1, 2]);  empty_396 = None
        empty_397: "bf16[512, 512, 49]" = torch.ops.aten.empty.memory_format([512, 512, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_130: "bf16[512, 512, 49]" = torch.ops.aten.permute.default(empty_397, [0, 1, 2]);  empty_397 = None
        triton_kernel_wrapper_functional_proxy_260 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 396, constant_args_idx = 571, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1351, 'MEAN': div_132, 'INVSTD': rsqrt_44, 'GAMMA': primals_268, 'BETA': primals_269, 'Y': permute_129, 'X_hat': permute_130, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_1351 = div_132 = primals_269 = permute_129 = permute_130 = None
        getitem_570: "bf16[512, 512, 49]" = triton_kernel_wrapper_functional_proxy_260['Y']
        getitem_571: "bf16[512, 512, 49]" = triton_kernel_wrapper_functional_proxy_260['X_hat'];  triton_kernel_wrapper_functional_proxy_260 = None
        empty_398: "i32[50176, 16]" = torch.ops.aten.empty.memory_format([50176, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_399: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_400: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1357: "bf16[512, 98, 256]" = torch.ops.aten.view.default(getitem_571, [512, -1, 256]);  getitem_571 = None
        view_1358: "bf16[50176, 256]" = torch.ops.aten.view.default(view_1357, [50176, 256]);  view_1357 = None
        triton_kernel_wrapper_functional_proxy_261 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 397, constant_args_idx = 572, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1358, 'P_ptr': empty_398, 'S_ptr': empty_399, 'M_ptr': empty_400, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1358 = empty_398 = empty_399 = empty_400 = None
        getitem_572: "i32[50176, 16]" = triton_kernel_wrapper_functional_proxy_261['P_ptr']
        getitem_573: "bf16[50176]" = triton_kernel_wrapper_functional_proxy_261['S_ptr']
        getitem_574: "bf16[50176]" = triton_kernel_wrapper_functional_proxy_261['M_ptr'];  triton_kernel_wrapper_functional_proxy_261 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_262: "i8[512, 512, 7, 7]" = torch.ops.aten.full.default([512, 512, 7, 7], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_1364: "bf16[512, 512, 7, 7]" = torch.ops.aten.view.default(getitem_570, [512, 512, 7, 7]);  getitem_570 = None
        empty_401: "bf16[512, 512, 7, 7]" = torch.ops.aten.empty.memory_format([512, 512, 7, 7], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_131: "bf16[512, 512, 7, 7]" = torch.ops.aten.permute.default(empty_401, [0, 1, 2, 3]);  empty_401 = None
        triton_kernel_wrapper_functional_proxy_262 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 573, grid = [(12544, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1364, 'Y_ptr': permute_131, 'Mask_prt': full_default_262, 'n_elts': 12845056, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_1364 = permute_131 = None
        getitem_575: "bf16[512, 512, 7, 7]" = triton_kernel_wrapper_functional_proxy_262['Y_ptr']
        getitem_576: "i8[512, 512, 7, 7]" = triton_kernel_wrapper_functional_proxy_262['Mask_prt'];  triton_kernel_wrapper_functional_proxy_262 = None
        full_default_263: "i32[50176, 8]" = torch.ops.aten.full.default([50176, 8], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_1369: "i8[512, 98, 256]" = torch.ops.aten.view.default(getitem_576, [512, -1, 256]);  getitem_576 = None
        view_1370: "i8[50176, 256]" = torch.ops.aten.view.default(view_1369, [50176, 256]);  view_1369 = None
        triton_kernel_wrapper_functional_proxy_263 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 574, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1370, 'P_ptr': full_default_263, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_1370 = None
        getitem_577: "i32[50176, 8]" = triton_kernel_wrapper_functional_proxy_263['P_ptr'];  triton_kernel_wrapper_functional_proxy_263 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_46: "bf16[2048, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_272, torch.bfloat16);  primals_272 = None
        empty_402: "i32[50176, 16]" = torch.ops.aten.empty.memory_format([50176, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_403: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_404: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1375: "bf16[512, 98, 256]" = torch.ops.aten.view.default(getitem_575, [512, -1, 256])
        view_1376: "bf16[50176, 256]" = torch.ops.aten.view.default(view_1375, [50176, 256]);  view_1375 = None
        triton_kernel_wrapper_functional_proxy_264 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 398, constant_args_idx = 575, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1376, 'P_ptr': empty_402, 'S_ptr': empty_403, 'M_ptr': empty_404, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1376 = empty_402 = empty_403 = empty_404 = None
        getitem_578: "i32[50176, 16]" = triton_kernel_wrapper_functional_proxy_264['P_ptr']
        getitem_579: "bf16[50176]" = triton_kernel_wrapper_functional_proxy_264['S_ptr']
        getitem_580: "bf16[50176]" = triton_kernel_wrapper_functional_proxy_264['M_ptr'];  triton_kernel_wrapper_functional_proxy_264 = None
        convolution_45: "bf16[512, 2048, 7, 7]" = torch.ops.aten.convolution.default(getitem_575, convert_element_type_46, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_575 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_193: "i64[]" = torch.ops.aten.add.Tensor(primals_273, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_1382: "bf16[512, 2048, 49]" = torch.ops.aten.view.default(convolution_45, [512, 2048, 49]);  convolution_45 = None
        full_default_264: "f32[2048]" = torch.ops.aten.full.default([2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_265 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 399, constant_args_idx = 576, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1382, 'SUM': full_default_264, 'SUMSQ': full_default_264, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_581: "f32[2048]" = triton_kernel_wrapper_functional_proxy_265['SUM']
        getitem_582: "f32[2048]" = triton_kernel_wrapper_functional_proxy_265['SUMSQ'];  triton_kernel_wrapper_functional_proxy_265 = None
        div_135: "f32[2048]" = torch.ops.aten.div.Tensor(getitem_581, full_default_260);  getitem_581 = None
        div_136: "f32[2048]" = torch.ops.aten.div.Tensor(getitem_582, full_default_260);  getitem_582 = None
        mul_316: "f32[2048]" = torch.ops.aten.mul.Tensor(div_135, div_135)
        sub_90: "f32[2048]" = torch.ops.aten.sub.Tensor(div_136, mul_316);  div_136 = mul_316 = None
        clamp_min_90: "f32[2048]" = torch.ops.aten.clamp_min.default(sub_90, 0.0);  sub_90 = None
        add_194: "f32[2048]" = torch.ops.aten.add.Tensor(clamp_min_90, 1e-05)
        rsqrt_45: "f32[2048]" = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
        mul_317: "f32[2048]" = torch.ops.aten.mul.Tensor(clamp_min_90, full_default_261);  clamp_min_90 = None
        mul_318: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_276, 0.9)
        mul_319: "f32[2048]" = torch.ops.aten.mul.Tensor(div_135, 0.1)
        add_195: "f32[2048]" = torch.ops.aten.add.Tensor(mul_318, mul_319);  mul_318 = mul_319 = None
        mul_320: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_277, 0.9)
        mul_321: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_317, 0.1);  mul_317 = None
        add_196: "f32[2048]" = torch.ops.aten.add.Tensor(mul_320, mul_321);  mul_320 = mul_321 = None
        empty_405: "bf16[512, 2048, 49]" = torch.ops.aten.empty.memory_format([512, 2048, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_132: "bf16[512, 2048, 49]" = torch.ops.aten.permute.default(empty_405, [0, 1, 2]);  empty_405 = None
        empty_406: "bf16[512, 2048, 49]" = torch.ops.aten.empty.memory_format([512, 2048, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_133: "bf16[512, 2048, 49]" = torch.ops.aten.permute.default(empty_406, [0, 1, 2]);  empty_406 = None
        triton_kernel_wrapper_functional_proxy_266 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 400, constant_args_idx = 577, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1382, 'MEAN': div_135, 'INVSTD': rsqrt_45, 'GAMMA': primals_274, 'BETA': primals_275, 'Y': permute_132, 'X_hat': permute_133, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_1382 = div_135 = primals_275 = permute_132 = permute_133 = None
        getitem_583: "bf16[512, 2048, 49]" = triton_kernel_wrapper_functional_proxy_266['Y']
        getitem_584: "bf16[512, 2048, 49]" = triton_kernel_wrapper_functional_proxy_266['X_hat'];  triton_kernel_wrapper_functional_proxy_266 = None
        empty_407: "i32[200704, 16]" = torch.ops.aten.empty.memory_format([200704, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_408: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_409: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1388: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_584, [512, -1, 256]);  getitem_584 = None
        view_1389: "bf16[200704, 256]" = torch.ops.aten.view.default(view_1388, [200704, 256]);  view_1388 = None
        triton_kernel_wrapper_functional_proxy_267 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 401, constant_args_idx = 578, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1389, 'P_ptr': empty_407, 'S_ptr': empty_408, 'M_ptr': empty_409, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1389 = empty_407 = empty_408 = empty_409 = None
        getitem_585: "i32[200704, 16]" = triton_kernel_wrapper_functional_proxy_267['P_ptr']
        getitem_586: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_267['S_ptr']
        getitem_587: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_267['M_ptr'];  triton_kernel_wrapper_functional_proxy_267 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_47: "bf16[2048, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_278, torch.bfloat16);  primals_278 = None
        empty_410: "i32[401408, 16]" = torch.ops.aten.empty.memory_format([401408, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_411: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_412: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_268 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 402, constant_args_idx = 579, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1314, 'P_ptr': empty_410, 'S_ptr': empty_411, 'M_ptr': empty_412, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1314 = empty_410 = empty_411 = empty_412 = None
        getitem_588: "i32[401408, 16]" = triton_kernel_wrapper_functional_proxy_268['P_ptr']
        getitem_589: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_268['S_ptr']
        getitem_590: "bf16[401408]" = triton_kernel_wrapper_functional_proxy_268['M_ptr'];  triton_kernel_wrapper_functional_proxy_268 = None
        convolution_46: "bf16[512, 2048, 7, 7]" = torch.ops.aten.convolution.default(getitem_549, convert_element_type_47, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  getitem_549 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_197: "i64[]" = torch.ops.aten.add.Tensor(primals_279, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_1406: "bf16[512, 2048, 49]" = torch.ops.aten.view.default(convolution_46, [512, 2048, 49]);  convolution_46 = None
        triton_kernel_wrapper_functional_proxy_269 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 403, constant_args_idx = 580, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1406, 'SUM': full_default_264, 'SUMSQ': full_default_264, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_591: "f32[2048]" = triton_kernel_wrapper_functional_proxy_269['SUM']
        getitem_592: "f32[2048]" = triton_kernel_wrapper_functional_proxy_269['SUMSQ'];  triton_kernel_wrapper_functional_proxy_269 = None
        div_138: "f32[2048]" = torch.ops.aten.div.Tensor(getitem_591, full_default_260);  getitem_591 = None
        div_139: "f32[2048]" = torch.ops.aten.div.Tensor(getitem_592, full_default_260);  getitem_592 = None
        mul_323: "f32[2048]" = torch.ops.aten.mul.Tensor(div_138, div_138)
        sub_92: "f32[2048]" = torch.ops.aten.sub.Tensor(div_139, mul_323);  div_139 = mul_323 = None
        clamp_min_92: "f32[2048]" = torch.ops.aten.clamp_min.default(sub_92, 0.0);  sub_92 = None
        add_198: "f32[2048]" = torch.ops.aten.add.Tensor(clamp_min_92, 1e-05)
        rsqrt_46: "f32[2048]" = torch.ops.aten.rsqrt.default(add_198);  add_198 = None
        mul_324: "f32[2048]" = torch.ops.aten.mul.Tensor(clamp_min_92, full_default_261);  clamp_min_92 = None
        mul_325: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_282, 0.9)
        mul_326: "f32[2048]" = torch.ops.aten.mul.Tensor(div_138, 0.1)
        add_199: "f32[2048]" = torch.ops.aten.add.Tensor(mul_325, mul_326);  mul_325 = mul_326 = None
        mul_327: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_283, 0.9)
        mul_328: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_324, 0.1);  mul_324 = None
        add_200: "f32[2048]" = torch.ops.aten.add.Tensor(mul_327, mul_328);  mul_327 = mul_328 = None
        empty_413: "bf16[512, 2048, 49]" = torch.ops.aten.empty.memory_format([512, 2048, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_134: "bf16[512, 2048, 49]" = torch.ops.aten.permute.default(empty_413, [0, 1, 2]);  empty_413 = None
        empty_414: "bf16[512, 2048, 49]" = torch.ops.aten.empty.memory_format([512, 2048, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_135: "bf16[512, 2048, 49]" = torch.ops.aten.permute.default(empty_414, [0, 1, 2]);  empty_414 = None
        triton_kernel_wrapper_functional_proxy_270 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 404, constant_args_idx = 581, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1406, 'MEAN': div_138, 'INVSTD': rsqrt_46, 'GAMMA': primals_280, 'BETA': primals_281, 'Y': permute_134, 'X_hat': permute_135, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_1406 = div_138 = primals_281 = permute_134 = permute_135 = None
        getitem_593: "bf16[512, 2048, 49]" = triton_kernel_wrapper_functional_proxy_270['Y']
        getitem_594: "bf16[512, 2048, 49]" = triton_kernel_wrapper_functional_proxy_270['X_hat'];  triton_kernel_wrapper_functional_proxy_270 = None
        empty_415: "i32[200704, 16]" = torch.ops.aten.empty.memory_format([200704, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_416: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_417: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1412: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_594, [512, -1, 256]);  getitem_594 = None
        view_1413: "bf16[200704, 256]" = torch.ops.aten.view.default(view_1412, [200704, 256]);  view_1412 = None
        triton_kernel_wrapper_functional_proxy_271 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 405, constant_args_idx = 582, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1413, 'P_ptr': empty_415, 'S_ptr': empty_416, 'M_ptr': empty_417, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1413 = empty_415 = empty_416 = empty_417 = None
        getitem_595: "i32[200704, 16]" = triton_kernel_wrapper_functional_proxy_271['P_ptr']
        getitem_596: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_271['S_ptr']
        getitem_597: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_271['M_ptr'];  triton_kernel_wrapper_functional_proxy_271 = None
        
         # File: <eval_with_key>.3 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:155 in forward, code: add_13 = layer4_0_bn3 + layer4_0_downsample_1;  layer4_0_bn3 = layer4_0_downsample_1 = None
        view_1419: "bf16[512, 2048, 7, 7]" = torch.ops.aten.view.default(getitem_583, [512, 2048, 7, 7]);  getitem_583 = None
        view_1420: "bf16[512, 2048, 7, 7]" = torch.ops.aten.view.default(getitem_593, [512, 2048, 7, 7]);  getitem_593 = None
        add_201: "bf16[512, 2048, 7, 7]" = torch.ops.aten.add.Tensor(view_1419, view_1420);  view_1419 = view_1420 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_272: "i8[512, 2048, 7, 7]" = torch.ops.aten.full.default([512, 2048, 7, 7], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        empty_418: "bf16[512, 2048, 7, 7]" = torch.ops.aten.empty.memory_format([512, 2048, 7, 7], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_136: "bf16[512, 2048, 7, 7]" = torch.ops.aten.permute.default(empty_418, [0, 1, 2, 3]);  empty_418 = None
        triton_kernel_wrapper_functional_proxy_272 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 583, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_201, 'Y_ptr': permute_136, 'Mask_prt': full_default_272, 'n_elts': 51380224, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  add_201 = permute_136 = None
        getitem_598: "bf16[512, 2048, 7, 7]" = triton_kernel_wrapper_functional_proxy_272['Y_ptr']
        getitem_599: "i8[512, 2048, 7, 7]" = triton_kernel_wrapper_functional_proxy_272['Mask_prt'];  triton_kernel_wrapper_functional_proxy_272 = None
        view_1425: "i8[512, 392, 256]" = torch.ops.aten.view.default(getitem_599, [512, -1, 256]);  getitem_599 = None
        view_1426: "i8[200704, 256]" = torch.ops.aten.view.default(view_1425, [200704, 256]);  view_1425 = None
        triton_kernel_wrapper_functional_proxy_273 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 584, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1426, 'P_ptr': full_default_75, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_1426 = None
        getitem_600: "i32[200704, 8]" = triton_kernel_wrapper_functional_proxy_273['P_ptr'];  triton_kernel_wrapper_functional_proxy_273 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_48: "bf16[512, 2048, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_284, torch.bfloat16);  primals_284 = None
        empty_419: "i32[200704, 16]" = torch.ops.aten.empty.memory_format([200704, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_420: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_421: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1431: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_598, [512, -1, 256])
        view_1432: "bf16[200704, 256]" = torch.ops.aten.view.default(view_1431, [200704, 256]);  view_1431 = None
        triton_kernel_wrapper_functional_proxy_274 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 406, constant_args_idx = 585, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1432, 'P_ptr': empty_419, 'S_ptr': empty_420, 'M_ptr': empty_421, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1432 = empty_419 = empty_420 = empty_421 = None
        getitem_601: "i32[200704, 16]" = triton_kernel_wrapper_functional_proxy_274['P_ptr']
        getitem_602: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_274['S_ptr']
        getitem_603: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_274['M_ptr'];  triton_kernel_wrapper_functional_proxy_274 = None
        convolution_47: "bf16[512, 512, 7, 7]" = torch.ops.aten.convolution.default(getitem_598, convert_element_type_48, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_202: "i64[]" = torch.ops.aten.add.Tensor(primals_285, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_1438: "bf16[512, 512, 49]" = torch.ops.aten.view.default(convolution_47, [512, 512, 49]);  convolution_47 = None
        triton_kernel_wrapper_functional_proxy_275 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 407, constant_args_idx = 586, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1438, 'SUM': full_default_76, 'SUMSQ': full_default_76, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_604: "f32[512]" = triton_kernel_wrapper_functional_proxy_275['SUM']
        getitem_605: "f32[512]" = triton_kernel_wrapper_functional_proxy_275['SUMSQ'];  triton_kernel_wrapper_functional_proxy_275 = None
        div_141: "f32[512]" = torch.ops.aten.div.Tensor(getitem_604, full_default_260);  getitem_604 = None
        div_142: "f32[512]" = torch.ops.aten.div.Tensor(getitem_605, full_default_260);  getitem_605 = None
        mul_330: "f32[512]" = torch.ops.aten.mul.Tensor(div_141, div_141)
        sub_94: "f32[512]" = torch.ops.aten.sub.Tensor(div_142, mul_330);  div_142 = mul_330 = None
        clamp_min_94: "f32[512]" = torch.ops.aten.clamp_min.default(sub_94, 0.0);  sub_94 = None
        add_203: "f32[512]" = torch.ops.aten.add.Tensor(clamp_min_94, 1e-05)
        rsqrt_47: "f32[512]" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
        mul_331: "f32[512]" = torch.ops.aten.mul.Tensor(clamp_min_94, full_default_261);  clamp_min_94 = None
        mul_332: "f32[512]" = torch.ops.aten.mul.Tensor(primals_288, 0.9)
        mul_333: "f32[512]" = torch.ops.aten.mul.Tensor(div_141, 0.1)
        add_204: "f32[512]" = torch.ops.aten.add.Tensor(mul_332, mul_333);  mul_332 = mul_333 = None
        mul_334: "f32[512]" = torch.ops.aten.mul.Tensor(primals_289, 0.9)
        mul_335: "f32[512]" = torch.ops.aten.mul.Tensor(mul_331, 0.1);  mul_331 = None
        add_205: "f32[512]" = torch.ops.aten.add.Tensor(mul_334, mul_335);  mul_334 = mul_335 = None
        empty_422: "bf16[512, 512, 49]" = torch.ops.aten.empty.memory_format([512, 512, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_137: "bf16[512, 512, 49]" = torch.ops.aten.permute.default(empty_422, [0, 1, 2]);  empty_422 = None
        empty_423: "bf16[512, 512, 49]" = torch.ops.aten.empty.memory_format([512, 512, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_138: "bf16[512, 512, 49]" = torch.ops.aten.permute.default(empty_423, [0, 1, 2]);  empty_423 = None
        triton_kernel_wrapper_functional_proxy_276 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 408, constant_args_idx = 587, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1438, 'MEAN': div_141, 'INVSTD': rsqrt_47, 'GAMMA': primals_286, 'BETA': primals_287, 'Y': permute_137, 'X_hat': permute_138, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_1438 = div_141 = primals_287 = permute_137 = permute_138 = None
        getitem_606: "bf16[512, 512, 49]" = triton_kernel_wrapper_functional_proxy_276['Y']
        getitem_607: "bf16[512, 512, 49]" = triton_kernel_wrapper_functional_proxy_276['X_hat'];  triton_kernel_wrapper_functional_proxy_276 = None
        empty_424: "i32[50176, 16]" = torch.ops.aten.empty.memory_format([50176, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_425: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_426: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1444: "bf16[512, 98, 256]" = torch.ops.aten.view.default(getitem_607, [512, -1, 256]);  getitem_607 = None
        view_1445: "bf16[50176, 256]" = torch.ops.aten.view.default(view_1444, [50176, 256]);  view_1444 = None
        triton_kernel_wrapper_functional_proxy_277 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 409, constant_args_idx = 588, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1445, 'P_ptr': empty_424, 'S_ptr': empty_425, 'M_ptr': empty_426, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1445 = empty_424 = empty_425 = empty_426 = None
        getitem_608: "i32[50176, 16]" = triton_kernel_wrapper_functional_proxy_277['P_ptr']
        getitem_609: "bf16[50176]" = triton_kernel_wrapper_functional_proxy_277['S_ptr']
        getitem_610: "bf16[50176]" = triton_kernel_wrapper_functional_proxy_277['M_ptr'];  triton_kernel_wrapper_functional_proxy_277 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1451: "bf16[512, 512, 7, 7]" = torch.ops.aten.view.default(getitem_606, [512, 512, 7, 7]);  getitem_606 = None
        empty_427: "bf16[512, 512, 7, 7]" = torch.ops.aten.empty.memory_format([512, 512, 7, 7], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_139: "bf16[512, 512, 7, 7]" = torch.ops.aten.permute.default(empty_427, [0, 1, 2, 3]);  empty_427 = None
        triton_kernel_wrapper_functional_proxy_278 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 589, grid = [(12544, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1451, 'Y_ptr': permute_139, 'Mask_prt': full_default_262, 'n_elts': 12845056, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_1451 = permute_139 = None
        getitem_611: "bf16[512, 512, 7, 7]" = triton_kernel_wrapper_functional_proxy_278['Y_ptr']
        getitem_612: "i8[512, 512, 7, 7]" = triton_kernel_wrapper_functional_proxy_278['Mask_prt'];  triton_kernel_wrapper_functional_proxy_278 = None
        view_1456: "i8[512, 98, 256]" = torch.ops.aten.view.default(getitem_612, [512, -1, 256]);  getitem_612 = None
        view_1457: "i8[50176, 256]" = torch.ops.aten.view.default(view_1456, [50176, 256]);  view_1456 = None
        triton_kernel_wrapper_functional_proxy_279 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 590, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1457, 'P_ptr': full_default_263, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_1457 = None
        getitem_613: "i32[50176, 8]" = triton_kernel_wrapper_functional_proxy_279['P_ptr'];  triton_kernel_wrapper_functional_proxy_279 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_49: "bf16[512, 512, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_290, torch.bfloat16);  primals_290 = None
        empty_428: "i32[50176, 16]" = torch.ops.aten.empty.memory_format([50176, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_429: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_430: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1462: "bf16[512, 98, 256]" = torch.ops.aten.view.default(getitem_611, [512, -1, 256])
        view_1463: "bf16[50176, 256]" = torch.ops.aten.view.default(view_1462, [50176, 256]);  view_1462 = None
        triton_kernel_wrapper_functional_proxy_280 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 410, constant_args_idx = 591, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1463, 'P_ptr': empty_428, 'S_ptr': empty_429, 'M_ptr': empty_430, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1463 = empty_428 = empty_429 = empty_430 = None
        getitem_614: "i32[50176, 16]" = triton_kernel_wrapper_functional_proxy_280['P_ptr']
        getitem_615: "bf16[50176]" = triton_kernel_wrapper_functional_proxy_280['S_ptr']
        getitem_616: "bf16[50176]" = triton_kernel_wrapper_functional_proxy_280['M_ptr'];  triton_kernel_wrapper_functional_proxy_280 = None
        convolution_48: "bf16[512, 512, 7, 7]" = torch.ops.aten.convolution.default(getitem_611, convert_element_type_49, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_611 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_206: "i64[]" = torch.ops.aten.add.Tensor(primals_291, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_1469: "bf16[512, 512, 49]" = torch.ops.aten.view.default(convolution_48, [512, 512, 49]);  convolution_48 = None
        triton_kernel_wrapper_functional_proxy_281 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 411, constant_args_idx = 592, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1469, 'SUM': full_default_76, 'SUMSQ': full_default_76, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_617: "f32[512]" = triton_kernel_wrapper_functional_proxy_281['SUM']
        getitem_618: "f32[512]" = triton_kernel_wrapper_functional_proxy_281['SUMSQ'];  triton_kernel_wrapper_functional_proxy_281 = None
        div_144: "f32[512]" = torch.ops.aten.div.Tensor(getitem_617, full_default_260);  getitem_617 = None
        div_145: "f32[512]" = torch.ops.aten.div.Tensor(getitem_618, full_default_260);  getitem_618 = None
        mul_337: "f32[512]" = torch.ops.aten.mul.Tensor(div_144, div_144)
        sub_96: "f32[512]" = torch.ops.aten.sub.Tensor(div_145, mul_337);  div_145 = mul_337 = None
        clamp_min_96: "f32[512]" = torch.ops.aten.clamp_min.default(sub_96, 0.0);  sub_96 = None
        add_207: "f32[512]" = torch.ops.aten.add.Tensor(clamp_min_96, 1e-05)
        rsqrt_48: "f32[512]" = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
        mul_338: "f32[512]" = torch.ops.aten.mul.Tensor(clamp_min_96, full_default_261);  clamp_min_96 = None
        mul_339: "f32[512]" = torch.ops.aten.mul.Tensor(primals_294, 0.9)
        mul_340: "f32[512]" = torch.ops.aten.mul.Tensor(div_144, 0.1)
        add_208: "f32[512]" = torch.ops.aten.add.Tensor(mul_339, mul_340);  mul_339 = mul_340 = None
        mul_341: "f32[512]" = torch.ops.aten.mul.Tensor(primals_295, 0.9)
        mul_342: "f32[512]" = torch.ops.aten.mul.Tensor(mul_338, 0.1);  mul_338 = None
        add_209: "f32[512]" = torch.ops.aten.add.Tensor(mul_341, mul_342);  mul_341 = mul_342 = None
        empty_431: "bf16[512, 512, 49]" = torch.ops.aten.empty.memory_format([512, 512, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_140: "bf16[512, 512, 49]" = torch.ops.aten.permute.default(empty_431, [0, 1, 2]);  empty_431 = None
        empty_432: "bf16[512, 512, 49]" = torch.ops.aten.empty.memory_format([512, 512, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_141: "bf16[512, 512, 49]" = torch.ops.aten.permute.default(empty_432, [0, 1, 2]);  empty_432 = None
        triton_kernel_wrapper_functional_proxy_282 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 412, constant_args_idx = 593, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1469, 'MEAN': div_144, 'INVSTD': rsqrt_48, 'GAMMA': primals_292, 'BETA': primals_293, 'Y': permute_140, 'X_hat': permute_141, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_1469 = div_144 = primals_293 = permute_140 = permute_141 = None
        getitem_619: "bf16[512, 512, 49]" = triton_kernel_wrapper_functional_proxy_282['Y']
        getitem_620: "bf16[512, 512, 49]" = triton_kernel_wrapper_functional_proxy_282['X_hat'];  triton_kernel_wrapper_functional_proxy_282 = None
        empty_433: "i32[50176, 16]" = torch.ops.aten.empty.memory_format([50176, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_434: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_435: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1475: "bf16[512, 98, 256]" = torch.ops.aten.view.default(getitem_620, [512, -1, 256]);  getitem_620 = None
        view_1476: "bf16[50176, 256]" = torch.ops.aten.view.default(view_1475, [50176, 256]);  view_1475 = None
        triton_kernel_wrapper_functional_proxy_283 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 413, constant_args_idx = 594, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1476, 'P_ptr': empty_433, 'S_ptr': empty_434, 'M_ptr': empty_435, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1476 = empty_433 = empty_434 = empty_435 = None
        getitem_621: "i32[50176, 16]" = triton_kernel_wrapper_functional_proxy_283['P_ptr']
        getitem_622: "bf16[50176]" = triton_kernel_wrapper_functional_proxy_283['S_ptr']
        getitem_623: "bf16[50176]" = triton_kernel_wrapper_functional_proxy_283['M_ptr'];  triton_kernel_wrapper_functional_proxy_283 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1482: "bf16[512, 512, 7, 7]" = torch.ops.aten.view.default(getitem_619, [512, 512, 7, 7]);  getitem_619 = None
        empty_436: "bf16[512, 512, 7, 7]" = torch.ops.aten.empty.memory_format([512, 512, 7, 7], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_142: "bf16[512, 512, 7, 7]" = torch.ops.aten.permute.default(empty_436, [0, 1, 2, 3]);  empty_436 = None
        triton_kernel_wrapper_functional_proxy_284 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 595, grid = [(12544, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1482, 'Y_ptr': permute_142, 'Mask_prt': full_default_262, 'n_elts': 12845056, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_1482 = permute_142 = None
        getitem_624: "bf16[512, 512, 7, 7]" = triton_kernel_wrapper_functional_proxy_284['Y_ptr']
        getitem_625: "i8[512, 512, 7, 7]" = triton_kernel_wrapper_functional_proxy_284['Mask_prt'];  triton_kernel_wrapper_functional_proxy_284 = None
        view_1487: "i8[512, 98, 256]" = torch.ops.aten.view.default(getitem_625, [512, -1, 256]);  getitem_625 = None
        view_1488: "i8[50176, 256]" = torch.ops.aten.view.default(view_1487, [50176, 256]);  view_1487 = None
        triton_kernel_wrapper_functional_proxy_285 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 596, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1488, 'P_ptr': full_default_263, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_1488 = None
        getitem_626: "i32[50176, 8]" = triton_kernel_wrapper_functional_proxy_285['P_ptr'];  triton_kernel_wrapper_functional_proxy_285 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_50: "bf16[2048, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_296, torch.bfloat16);  primals_296 = None
        empty_437: "i32[50176, 16]" = torch.ops.aten.empty.memory_format([50176, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_438: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_439: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1493: "bf16[512, 98, 256]" = torch.ops.aten.view.default(getitem_624, [512, -1, 256])
        view_1494: "bf16[50176, 256]" = torch.ops.aten.view.default(view_1493, [50176, 256]);  view_1493 = None
        triton_kernel_wrapper_functional_proxy_286 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 414, constant_args_idx = 597, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1494, 'P_ptr': empty_437, 'S_ptr': empty_438, 'M_ptr': empty_439, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1494 = empty_437 = empty_438 = empty_439 = None
        getitem_627: "i32[50176, 16]" = triton_kernel_wrapper_functional_proxy_286['P_ptr']
        getitem_628: "bf16[50176]" = triton_kernel_wrapper_functional_proxy_286['S_ptr']
        getitem_629: "bf16[50176]" = triton_kernel_wrapper_functional_proxy_286['M_ptr'];  triton_kernel_wrapper_functional_proxy_286 = None
        convolution_49: "bf16[512, 2048, 7, 7]" = torch.ops.aten.convolution.default(getitem_624, convert_element_type_50, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_624 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_210: "i64[]" = torch.ops.aten.add.Tensor(primals_297, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_1500: "bf16[512, 2048, 49]" = torch.ops.aten.view.default(convolution_49, [512, 2048, 49]);  convolution_49 = None
        triton_kernel_wrapper_functional_proxy_287 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 415, constant_args_idx = 598, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1500, 'SUM': full_default_264, 'SUMSQ': full_default_264, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_630: "f32[2048]" = triton_kernel_wrapper_functional_proxy_287['SUM']
        getitem_631: "f32[2048]" = triton_kernel_wrapper_functional_proxy_287['SUMSQ'];  triton_kernel_wrapper_functional_proxy_287 = None
        div_147: "f32[2048]" = torch.ops.aten.div.Tensor(getitem_630, full_default_260);  getitem_630 = None
        div_148: "f32[2048]" = torch.ops.aten.div.Tensor(getitem_631, full_default_260);  getitem_631 = None
        mul_344: "f32[2048]" = torch.ops.aten.mul.Tensor(div_147, div_147)
        sub_98: "f32[2048]" = torch.ops.aten.sub.Tensor(div_148, mul_344);  div_148 = mul_344 = None
        clamp_min_98: "f32[2048]" = torch.ops.aten.clamp_min.default(sub_98, 0.0);  sub_98 = None
        add_211: "f32[2048]" = torch.ops.aten.add.Tensor(clamp_min_98, 1e-05)
        rsqrt_49: "f32[2048]" = torch.ops.aten.rsqrt.default(add_211);  add_211 = None
        mul_345: "f32[2048]" = torch.ops.aten.mul.Tensor(clamp_min_98, full_default_261);  clamp_min_98 = None
        mul_346: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_300, 0.9)
        mul_347: "f32[2048]" = torch.ops.aten.mul.Tensor(div_147, 0.1)
        add_212: "f32[2048]" = torch.ops.aten.add.Tensor(mul_346, mul_347);  mul_346 = mul_347 = None
        mul_348: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_301, 0.9)
        mul_349: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_345, 0.1);  mul_345 = None
        add_213: "f32[2048]" = torch.ops.aten.add.Tensor(mul_348, mul_349);  mul_348 = mul_349 = None
        empty_440: "bf16[512, 2048, 49]" = torch.ops.aten.empty.memory_format([512, 2048, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_143: "bf16[512, 2048, 49]" = torch.ops.aten.permute.default(empty_440, [0, 1, 2]);  empty_440 = None
        empty_441: "bf16[512, 2048, 49]" = torch.ops.aten.empty.memory_format([512, 2048, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_144: "bf16[512, 2048, 49]" = torch.ops.aten.permute.default(empty_441, [0, 1, 2]);  empty_441 = None
        triton_kernel_wrapper_functional_proxy_288 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 416, constant_args_idx = 599, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1500, 'MEAN': div_147, 'INVSTD': rsqrt_49, 'GAMMA': primals_298, 'BETA': primals_299, 'Y': permute_143, 'X_hat': permute_144, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_1500 = div_147 = primals_299 = permute_143 = permute_144 = None
        getitem_632: "bf16[512, 2048, 49]" = triton_kernel_wrapper_functional_proxy_288['Y']
        getitem_633: "bf16[512, 2048, 49]" = triton_kernel_wrapper_functional_proxy_288['X_hat'];  triton_kernel_wrapper_functional_proxy_288 = None
        empty_442: "i32[200704, 16]" = torch.ops.aten.empty.memory_format([200704, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_443: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_444: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1506: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_633, [512, -1, 256]);  getitem_633 = None
        view_1507: "bf16[200704, 256]" = torch.ops.aten.view.default(view_1506, [200704, 256]);  view_1506 = None
        triton_kernel_wrapper_functional_proxy_289 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 417, constant_args_idx = 600, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1507, 'P_ptr': empty_442, 'S_ptr': empty_443, 'M_ptr': empty_444, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1507 = empty_442 = empty_443 = empty_444 = None
        getitem_634: "i32[200704, 16]" = triton_kernel_wrapper_functional_proxy_289['P_ptr']
        getitem_635: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_289['S_ptr']
        getitem_636: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_289['M_ptr'];  triton_kernel_wrapper_functional_proxy_289 = None
        
         # File: <eval_with_key>.3 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:165 in forward, code: add_14 = layer4_1_bn3 + layer4_0_relu_2;  layer4_1_bn3 = layer4_0_relu_2 = None
        view_1513: "bf16[512, 2048, 7, 7]" = torch.ops.aten.view.default(getitem_632, [512, 2048, 7, 7]);  getitem_632 = None
        add_214: "bf16[512, 2048, 7, 7]" = torch.ops.aten.add.Tensor(view_1513, getitem_598);  view_1513 = getitem_598 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        empty_445: "bf16[512, 2048, 7, 7]" = torch.ops.aten.empty.memory_format([512, 2048, 7, 7], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_145: "bf16[512, 2048, 7, 7]" = torch.ops.aten.permute.default(empty_445, [0, 1, 2, 3]);  empty_445 = None
        triton_kernel_wrapper_functional_proxy_290 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 601, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_214, 'Y_ptr': permute_145, 'Mask_prt': full_default_272, 'n_elts': 51380224, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  add_214 = permute_145 = None
        getitem_637: "bf16[512, 2048, 7, 7]" = triton_kernel_wrapper_functional_proxy_290['Y_ptr']
        getitem_638: "i8[512, 2048, 7, 7]" = triton_kernel_wrapper_functional_proxy_290['Mask_prt'];  triton_kernel_wrapper_functional_proxy_290 = None
        view_1518: "i8[512, 392, 256]" = torch.ops.aten.view.default(getitem_638, [512, -1, 256]);  getitem_638 = None
        view_1519: "i8[200704, 256]" = torch.ops.aten.view.default(view_1518, [200704, 256]);  view_1518 = None
        triton_kernel_wrapper_functional_proxy_291 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 602, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1519, 'P_ptr': full_default_75, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_1519 = None
        getitem_639: "i32[200704, 8]" = triton_kernel_wrapper_functional_proxy_291['P_ptr'];  triton_kernel_wrapper_functional_proxy_291 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_51: "bf16[512, 2048, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_302, torch.bfloat16);  primals_302 = None
        empty_446: "i32[200704, 16]" = torch.ops.aten.empty.memory_format([200704, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_447: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_448: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1524: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_637, [512, -1, 256])
        view_1525: "bf16[200704, 256]" = torch.ops.aten.view.default(view_1524, [200704, 256]);  view_1524 = None
        triton_kernel_wrapper_functional_proxy_292 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 418, constant_args_idx = 603, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1525, 'P_ptr': empty_446, 'S_ptr': empty_447, 'M_ptr': empty_448, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1525 = empty_446 = empty_447 = empty_448 = None
        getitem_640: "i32[200704, 16]" = triton_kernel_wrapper_functional_proxy_292['P_ptr']
        getitem_641: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_292['S_ptr']
        getitem_642: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_292['M_ptr'];  triton_kernel_wrapper_functional_proxy_292 = None
        convolution_50: "bf16[512, 512, 7, 7]" = torch.ops.aten.convolution.default(getitem_637, convert_element_type_51, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_215: "i64[]" = torch.ops.aten.add.Tensor(primals_303, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_1531: "bf16[512, 512, 49]" = torch.ops.aten.view.default(convolution_50, [512, 512, 49]);  convolution_50 = None
        triton_kernel_wrapper_functional_proxy_293 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 419, constant_args_idx = 604, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1531, 'SUM': full_default_76, 'SUMSQ': full_default_76, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ'])
        getitem_643: "f32[512]" = triton_kernel_wrapper_functional_proxy_293['SUM']
        getitem_644: "f32[512]" = triton_kernel_wrapper_functional_proxy_293['SUMSQ'];  triton_kernel_wrapper_functional_proxy_293 = None
        div_150: "f32[512]" = torch.ops.aten.div.Tensor(getitem_643, full_default_260);  getitem_643 = None
        div_151: "f32[512]" = torch.ops.aten.div.Tensor(getitem_644, full_default_260);  getitem_644 = None
        mul_351: "f32[512]" = torch.ops.aten.mul.Tensor(div_150, div_150)
        sub_100: "f32[512]" = torch.ops.aten.sub.Tensor(div_151, mul_351);  div_151 = mul_351 = None
        clamp_min_100: "f32[512]" = torch.ops.aten.clamp_min.default(sub_100, 0.0);  sub_100 = None
        add_216: "f32[512]" = torch.ops.aten.add.Tensor(clamp_min_100, 1e-05)
        rsqrt_50: "f32[512]" = torch.ops.aten.rsqrt.default(add_216);  add_216 = None
        mul_352: "f32[512]" = torch.ops.aten.mul.Tensor(clamp_min_100, full_default_261);  clamp_min_100 = None
        mul_353: "f32[512]" = torch.ops.aten.mul.Tensor(primals_306, 0.9)
        mul_354: "f32[512]" = torch.ops.aten.mul.Tensor(div_150, 0.1)
        add_217: "f32[512]" = torch.ops.aten.add.Tensor(mul_353, mul_354);  mul_353 = mul_354 = None
        mul_355: "f32[512]" = torch.ops.aten.mul.Tensor(primals_307, 0.9)
        mul_356: "f32[512]" = torch.ops.aten.mul.Tensor(mul_352, 0.1);  mul_352 = None
        add_218: "f32[512]" = torch.ops.aten.add.Tensor(mul_355, mul_356);  mul_355 = mul_356 = None
        empty_449: "bf16[512, 512, 49]" = torch.ops.aten.empty.memory_format([512, 512, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_146: "bf16[512, 512, 49]" = torch.ops.aten.permute.default(empty_449, [0, 1, 2]);  empty_449 = None
        empty_450: "bf16[512, 512, 49]" = torch.ops.aten.empty.memory_format([512, 512, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_147: "bf16[512, 512, 49]" = torch.ops.aten.permute.default(empty_450, [0, 1, 2]);  empty_450 = None
        triton_kernel_wrapper_functional_proxy_294 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 420, constant_args_idx = 605, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1531, 'MEAN': div_150, 'INVSTD': rsqrt_50, 'GAMMA': primals_304, 'BETA': primals_305, 'Y': permute_146, 'X_hat': permute_147, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_1531 = div_150 = primals_305 = permute_146 = permute_147 = None
        getitem_645: "bf16[512, 512, 49]" = triton_kernel_wrapper_functional_proxy_294['Y']
        getitem_646: "bf16[512, 512, 49]" = triton_kernel_wrapper_functional_proxy_294['X_hat'];  triton_kernel_wrapper_functional_proxy_294 = None
        empty_451: "i32[50176, 16]" = torch.ops.aten.empty.memory_format([50176, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_452: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_453: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1537: "bf16[512, 98, 256]" = torch.ops.aten.view.default(getitem_646, [512, -1, 256]);  getitem_646 = None
        view_1538: "bf16[50176, 256]" = torch.ops.aten.view.default(view_1537, [50176, 256]);  view_1537 = None
        triton_kernel_wrapper_functional_proxy_295 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 421, constant_args_idx = 606, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1538, 'P_ptr': empty_451, 'S_ptr': empty_452, 'M_ptr': empty_453, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1538 = empty_451 = empty_452 = empty_453 = None
        getitem_647: "i32[50176, 16]" = triton_kernel_wrapper_functional_proxy_295['P_ptr']
        getitem_648: "bf16[50176]" = triton_kernel_wrapper_functional_proxy_295['S_ptr']
        getitem_649: "bf16[50176]" = triton_kernel_wrapper_functional_proxy_295['M_ptr'];  triton_kernel_wrapper_functional_proxy_295 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1544: "bf16[512, 512, 7, 7]" = torch.ops.aten.view.default(getitem_645, [512, 512, 7, 7]);  getitem_645 = None
        empty_454: "bf16[512, 512, 7, 7]" = torch.ops.aten.empty.memory_format([512, 512, 7, 7], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_148: "bf16[512, 512, 7, 7]" = torch.ops.aten.permute.default(empty_454, [0, 1, 2, 3]);  empty_454 = None
        triton_kernel_wrapper_functional_proxy_296 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 607, grid = [(12544, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1544, 'Y_ptr': permute_148, 'Mask_prt': full_default_262, 'n_elts': 12845056, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_1544 = permute_148 = None
        getitem_650: "bf16[512, 512, 7, 7]" = triton_kernel_wrapper_functional_proxy_296['Y_ptr']
        getitem_651: "i8[512, 512, 7, 7]" = triton_kernel_wrapper_functional_proxy_296['Mask_prt'];  triton_kernel_wrapper_functional_proxy_296 = None
        view_1549: "i8[512, 98, 256]" = torch.ops.aten.view.default(getitem_651, [512, -1, 256]);  getitem_651 = None
        view_1550: "i8[50176, 256]" = torch.ops.aten.view.default(view_1549, [50176, 256]);  view_1549 = None
        triton_kernel_wrapper_functional_proxy_297 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 608, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1550, 'P_ptr': full_default_263, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_1550 = None
        getitem_652: "i32[50176, 8]" = triton_kernel_wrapper_functional_proxy_297['P_ptr'];  triton_kernel_wrapper_functional_proxy_297 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_52: "bf16[512, 512, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_308, torch.bfloat16);  primals_308 = None
        empty_455: "i32[50176, 16]" = torch.ops.aten.empty.memory_format([50176, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_456: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_457: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1555: "bf16[512, 98, 256]" = torch.ops.aten.view.default(getitem_650, [512, -1, 256])
        view_1556: "bf16[50176, 256]" = torch.ops.aten.view.default(view_1555, [50176, 256]);  view_1555 = None
        triton_kernel_wrapper_functional_proxy_298 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 422, constant_args_idx = 609, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1556, 'P_ptr': empty_455, 'S_ptr': empty_456, 'M_ptr': empty_457, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1556 = empty_455 = empty_456 = empty_457 = None
        getitem_653: "i32[50176, 16]" = triton_kernel_wrapper_functional_proxy_298['P_ptr']
        getitem_654: "bf16[50176]" = triton_kernel_wrapper_functional_proxy_298['S_ptr']
        getitem_655: "bf16[50176]" = triton_kernel_wrapper_functional_proxy_298['M_ptr'];  triton_kernel_wrapper_functional_proxy_298 = None
        convolution_51: "bf16[512, 512, 7, 7]" = torch.ops.aten.convolution.default(getitem_650, convert_element_type_52, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_650 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_219: "i64[]" = torch.ops.aten.add.Tensor(primals_309, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_1562: "bf16[512, 512, 49]" = torch.ops.aten.view.default(convolution_51, [512, 512, 49]);  convolution_51 = None
        triton_kernel_wrapper_functional_proxy_299 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 423, constant_args_idx = 610, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1562, 'SUM': full_default_76, 'SUMSQ': full_default_76, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ']);  full_default_76 = None
        getitem_656: "f32[512]" = triton_kernel_wrapper_functional_proxy_299['SUM']
        getitem_657: "f32[512]" = triton_kernel_wrapper_functional_proxy_299['SUMSQ'];  triton_kernel_wrapper_functional_proxy_299 = None
        div_153: "f32[512]" = torch.ops.aten.div.Tensor(getitem_656, full_default_260);  getitem_656 = None
        div_154: "f32[512]" = torch.ops.aten.div.Tensor(getitem_657, full_default_260);  getitem_657 = None
        mul_358: "f32[512]" = torch.ops.aten.mul.Tensor(div_153, div_153)
        sub_102: "f32[512]" = torch.ops.aten.sub.Tensor(div_154, mul_358);  div_154 = mul_358 = None
        clamp_min_102: "f32[512]" = torch.ops.aten.clamp_min.default(sub_102, 0.0);  sub_102 = None
        add_220: "f32[512]" = torch.ops.aten.add.Tensor(clamp_min_102, 1e-05)
        rsqrt_51: "f32[512]" = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
        mul_359: "f32[512]" = torch.ops.aten.mul.Tensor(clamp_min_102, full_default_261);  clamp_min_102 = None
        mul_360: "f32[512]" = torch.ops.aten.mul.Tensor(primals_312, 0.9)
        mul_361: "f32[512]" = torch.ops.aten.mul.Tensor(div_153, 0.1)
        add_221: "f32[512]" = torch.ops.aten.add.Tensor(mul_360, mul_361);  mul_360 = mul_361 = None
        mul_362: "f32[512]" = torch.ops.aten.mul.Tensor(primals_313, 0.9)
        mul_363: "f32[512]" = torch.ops.aten.mul.Tensor(mul_359, 0.1);  mul_359 = None
        add_222: "f32[512]" = torch.ops.aten.add.Tensor(mul_362, mul_363);  mul_362 = mul_363 = None
        empty_458: "bf16[512, 512, 49]" = torch.ops.aten.empty.memory_format([512, 512, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_149: "bf16[512, 512, 49]" = torch.ops.aten.permute.default(empty_458, [0, 1, 2]);  empty_458 = None
        empty_459: "bf16[512, 512, 49]" = torch.ops.aten.empty.memory_format([512, 512, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_150: "bf16[512, 512, 49]" = torch.ops.aten.permute.default(empty_459, [0, 1, 2]);  empty_459 = None
        triton_kernel_wrapper_functional_proxy_300 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 424, constant_args_idx = 611, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1562, 'MEAN': div_153, 'INVSTD': rsqrt_51, 'GAMMA': primals_310, 'BETA': primals_311, 'Y': permute_149, 'X_hat': permute_150, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_1562 = div_153 = primals_311 = permute_149 = permute_150 = None
        getitem_658: "bf16[512, 512, 49]" = triton_kernel_wrapper_functional_proxy_300['Y']
        getitem_659: "bf16[512, 512, 49]" = triton_kernel_wrapper_functional_proxy_300['X_hat'];  triton_kernel_wrapper_functional_proxy_300 = None
        empty_460: "i32[50176, 16]" = torch.ops.aten.empty.memory_format([50176, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_461: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_462: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1568: "bf16[512, 98, 256]" = torch.ops.aten.view.default(getitem_659, [512, -1, 256]);  getitem_659 = None
        view_1569: "bf16[50176, 256]" = torch.ops.aten.view.default(view_1568, [50176, 256]);  view_1568 = None
        triton_kernel_wrapper_functional_proxy_301 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 425, constant_args_idx = 612, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1569, 'P_ptr': empty_460, 'S_ptr': empty_461, 'M_ptr': empty_462, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1569 = empty_460 = empty_461 = empty_462 = None
        getitem_660: "i32[50176, 16]" = triton_kernel_wrapper_functional_proxy_301['P_ptr']
        getitem_661: "bf16[50176]" = triton_kernel_wrapper_functional_proxy_301['S_ptr']
        getitem_662: "bf16[50176]" = triton_kernel_wrapper_functional_proxy_301['M_ptr'];  triton_kernel_wrapper_functional_proxy_301 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1575: "bf16[512, 512, 7, 7]" = torch.ops.aten.view.default(getitem_658, [512, 512, 7, 7]);  getitem_658 = None
        empty_463: "bf16[512, 512, 7, 7]" = torch.ops.aten.empty.memory_format([512, 512, 7, 7], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_151: "bf16[512, 512, 7, 7]" = torch.ops.aten.permute.default(empty_463, [0, 1, 2, 3]);  empty_463 = None
        triton_kernel_wrapper_functional_proxy_302 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 613, grid = [(12544, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1575, 'Y_ptr': permute_151, 'Mask_prt': full_default_262, 'n_elts': 12845056, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  view_1575 = permute_151 = full_default_262 = None
        getitem_663: "bf16[512, 512, 7, 7]" = triton_kernel_wrapper_functional_proxy_302['Y_ptr']
        getitem_664: "i8[512, 512, 7, 7]" = triton_kernel_wrapper_functional_proxy_302['Mask_prt'];  triton_kernel_wrapper_functional_proxy_302 = None
        view_1580: "i8[512, 98, 256]" = torch.ops.aten.view.default(getitem_664, [512, -1, 256]);  getitem_664 = None
        view_1581: "i8[50176, 256]" = torch.ops.aten.view.default(view_1580, [50176, 256]);  view_1580 = None
        triton_kernel_wrapper_functional_proxy_303 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 614, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1581, 'P_ptr': full_default_263, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_1581 = full_default_263 = None
        getitem_665: "i32[50176, 8]" = triton_kernel_wrapper_functional_proxy_303['P_ptr'];  triton_kernel_wrapper_functional_proxy_303 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_53: "bf16[2048, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_314, torch.bfloat16);  primals_314 = None
        empty_464: "i32[50176, 16]" = torch.ops.aten.empty.memory_format([50176, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_465: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_466: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1586: "bf16[512, 98, 256]" = torch.ops.aten.view.default(getitem_663, [512, -1, 256])
        view_1587: "bf16[50176, 256]" = torch.ops.aten.view.default(view_1586, [50176, 256]);  view_1586 = None
        triton_kernel_wrapper_functional_proxy_304 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 426, constant_args_idx = 615, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1587, 'P_ptr': empty_464, 'S_ptr': empty_465, 'M_ptr': empty_466, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1587 = empty_464 = empty_465 = empty_466 = None
        getitem_666: "i32[50176, 16]" = triton_kernel_wrapper_functional_proxy_304['P_ptr']
        getitem_667: "bf16[50176]" = triton_kernel_wrapper_functional_proxy_304['S_ptr']
        getitem_668: "bf16[50176]" = triton_kernel_wrapper_functional_proxy_304['M_ptr'];  triton_kernel_wrapper_functional_proxy_304 = None
        convolution_52: "bf16[512, 2048, 7, 7]" = torch.ops.aten.convolution.default(getitem_663, convert_element_type_53, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_663 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:180 in forward, code: self.num_batches_tracked.add_(1)
        add_223: "i64[]" = torch.ops.aten.add.Tensor(primals_315, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        view_1593: "bf16[512, 2048, 49]" = torch.ops.aten.view.default(convolution_52, [512, 2048, 49]);  convolution_52 = None
        triton_kernel_wrapper_functional_proxy_305 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 427, constant_args_idx = 616, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1593, 'SUM': full_default_264, 'SUMSQ': full_default_264, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['SUM', 'SUMSQ']);  full_default_264 = None
        getitem_669: "f32[2048]" = triton_kernel_wrapper_functional_proxy_305['SUM']
        getitem_670: "f32[2048]" = triton_kernel_wrapper_functional_proxy_305['SUMSQ'];  triton_kernel_wrapper_functional_proxy_305 = None
        div_156: "f32[2048]" = torch.ops.aten.div.Tensor(getitem_669, full_default_260);  getitem_669 = None
        div_157: "f32[2048]" = torch.ops.aten.div.Tensor(getitem_670, full_default_260);  getitem_670 = full_default_260 = None
        mul_365: "f32[2048]" = torch.ops.aten.mul.Tensor(div_156, div_156)
        sub_104: "f32[2048]" = torch.ops.aten.sub.Tensor(div_157, mul_365);  div_157 = mul_365 = None
        clamp_min_104: "f32[2048]" = torch.ops.aten.clamp_min.default(sub_104, 0.0);  sub_104 = None
        add_224: "f32[2048]" = torch.ops.aten.add.Tensor(clamp_min_104, 1e-05)
        rsqrt_52: "f32[2048]" = torch.ops.aten.rsqrt.default(add_224);  add_224 = None
        mul_366: "f32[2048]" = torch.ops.aten.mul.Tensor(clamp_min_104, full_default_261);  clamp_min_104 = full_default_261 = None
        mul_367: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_318, 0.9)
        mul_368: "f32[2048]" = torch.ops.aten.mul.Tensor(div_156, 0.1)
        add_225: "f32[2048]" = torch.ops.aten.add.Tensor(mul_367, mul_368);  mul_367 = mul_368 = None
        mul_369: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_319, 0.9)
        mul_370: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_366, 0.1);  mul_366 = None
        add_226: "f32[2048]" = torch.ops.aten.add.Tensor(mul_369, mul_370);  mul_369 = mul_370 = None
        empty_467: "bf16[512, 2048, 49]" = torch.ops.aten.empty.memory_format([512, 2048, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_152: "bf16[512, 2048, 49]" = torch.ops.aten.permute.default(empty_467, [0, 1, 2]);  empty_467 = None
        empty_468: "bf16[512, 2048, 49]" = torch.ops.aten.empty.memory_format([512, 2048, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_153: "bf16[512, 2048, 49]" = torch.ops.aten.permute.default(empty_468, [0, 1, 2]);  empty_468 = None
        triton_kernel_wrapper_functional_proxy_306 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 428, constant_args_idx = 617, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1593, 'MEAN': div_156, 'INVSTD': rsqrt_52, 'GAMMA': primals_316, 'BETA': primals_317, 'Y': permute_152, 'X_hat': permute_153, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['Y', 'X_hat']);  view_1593 = div_156 = primals_317 = permute_152 = permute_153 = None
        getitem_671: "bf16[512, 2048, 49]" = triton_kernel_wrapper_functional_proxy_306['Y']
        getitem_672: "bf16[512, 2048, 49]" = triton_kernel_wrapper_functional_proxy_306['X_hat'];  triton_kernel_wrapper_functional_proxy_306 = None
        empty_469: "i32[200704, 16]" = torch.ops.aten.empty.memory_format([200704, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_470: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_471: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1599: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_672, [512, -1, 256]);  getitem_672 = None
        view_1600: "bf16[200704, 256]" = torch.ops.aten.view.default(view_1599, [200704, 256]);  view_1599 = None
        triton_kernel_wrapper_functional_proxy_307 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 429, constant_args_idx = 618, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1600, 'P_ptr': empty_469, 'S_ptr': empty_470, 'M_ptr': empty_471, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1600 = empty_469 = empty_470 = empty_471 = None
        getitem_673: "i32[200704, 16]" = triton_kernel_wrapper_functional_proxy_307['P_ptr']
        getitem_674: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_307['S_ptr']
        getitem_675: "bf16[200704]" = triton_kernel_wrapper_functional_proxy_307['M_ptr'];  triton_kernel_wrapper_functional_proxy_307 = None
        
         # File: <eval_with_key>.3 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:175 in forward, code: add_15 = layer4_2_bn3 + layer4_1_relu_2;  layer4_2_bn3 = layer4_1_relu_2 = None
        view_1606: "bf16[512, 2048, 7, 7]" = torch.ops.aten.view.default(getitem_671, [512, 2048, 7, 7]);  getitem_671 = None
        add_227: "bf16[512, 2048, 7, 7]" = torch.ops.aten.add.Tensor(view_1606, getitem_637);  view_1606 = getitem_637 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        empty_472: "bf16[512, 2048, 7, 7]" = torch.ops.aten.empty.memory_format([512, 2048, 7, 7], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_154: "bf16[512, 2048, 7, 7]" = torch.ops.aten.permute.default(empty_472, [0, 1, 2, 3]);  empty_472 = None
        triton_kernel_wrapper_functional_proxy_308 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 7, constant_args_idx = 619, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_227, 'Y_ptr': permute_154, 'Mask_prt': full_default_272, 'n_elts': 51380224, 'BLOCK_SIZE': 1024}, tensors_to_clone = ['Y_ptr', 'Mask_prt']);  add_227 = permute_154 = full_default_272 = None
        getitem_676: "bf16[512, 2048, 7, 7]" = triton_kernel_wrapper_functional_proxy_308['Y_ptr']
        getitem_677: "i8[512, 2048, 7, 7]" = triton_kernel_wrapper_functional_proxy_308['Mask_prt'];  triton_kernel_wrapper_functional_proxy_308 = None
        view_1611: "i8[512, 392, 256]" = torch.ops.aten.view.default(getitem_677, [512, -1, 256]);  getitem_677 = None
        view_1612: "i8[200704, 256]" = torch.ops.aten.view.default(view_1611, [200704, 256]);  view_1611 = None
        triton_kernel_wrapper_functional_proxy_309 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 8, constant_args_idx = 620, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1612, 'P_ptr': full_default_75, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 8, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['P_ptr']);  view_1612 = full_default_75 = None
        getitem_678: "i32[200704, 8]" = triton_kernel_wrapper_functional_proxy_309['P_ptr'];  triton_kernel_wrapper_functional_proxy_309 = None
        
         # File: <eval_with_key>.3 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:177 in forward, code: avgpool = self.avgpool(layer4_2_relu_2);  layer4_2_relu_2 = None
        mean: "bf16[512, 2048, 1, 1]" = torch.ops.aten.mean.dim(getitem_676, [-1, -2], True);  getitem_676 = None
        
         # File: <eval_with_key>.3 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:178 in forward, code: flatten = torch.flatten(avgpool, 1);  avgpool = None
        view_1613: "bf16[512, 2048]" = torch.ops.aten.view.default(mean, [512, 2048]);  mean = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:26 in forward, code: return _QuanLinear.apply(x, self.weight, self.quantizer, self.target_name, self.graph_mode, self.meta, self.qdrop)
        view_1616: "bf16[512, 8, 256]" = torch.ops.aten.view.default(view_1613, [512, -1, 256])
        view_1617: "bf16[4096, 256]" = torch.ops.aten.view.default(view_1616, [4096, 256]);  view_1616 = None
        empty_473: "i32[4096, 16]" = torch.ops.aten.empty.memory_format([4096, 16], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_474: "bf16[4096]" = torch.ops.aten.empty.memory_format([4096], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_475: "bf16[4096]" = torch.ops.aten.empty.memory_format([4096], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_310 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 430, constant_args_idx = 621, grid = [(4096, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1617, 'P_ptr': empty_473, 'S_ptr': empty_474, 'M_ptr': empty_475, 'stride_x0': 256, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16, 'QMAX': 3}, tensors_to_clone = ['P_ptr', 'S_ptr', 'M_ptr']);  view_1617 = empty_473 = empty_474 = empty_475 = None
        getitem_679: "i32[4096, 16]" = triton_kernel_wrapper_functional_proxy_310['P_ptr']
        getitem_680: "bf16[4096]" = triton_kernel_wrapper_functional_proxy_310['S_ptr']
        getitem_681: "bf16[4096]" = triton_kernel_wrapper_functional_proxy_310['M_ptr'];  triton_kernel_wrapper_functional_proxy_310 = None
        convert_element_type_54: "bf16[100, 2048]" = torch.ops.prims.convert_element_type.default(primals_320, torch.bfloat16);  primals_320 = None
        permute_155: "bf16[2048, 100]" = torch.ops.aten.permute.default(convert_element_type_54, [1, 0])
        mm: "bf16[512, 100]" = torch.ops.aten.mm.default(view_1613, permute_155);  view_1613 = permute_155 = None
        
         # File: /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torch/__init__.py:2380 in __call__, code: return compile_fx(model_, inputs_, config_patches=self.config)
        copy_: "i64[]" = torch.ops.aten.copy_.default(primals_3, add);  primals_3 = add = copy_ = None
        copy__1: "f32[64]" = torch.ops.aten.copy_.default(primals_6, add_2);  primals_6 = add_2 = copy__1 = None
        copy__2: "f32[64]" = torch.ops.aten.copy_.default(primals_7, add_3);  primals_7 = add_3 = copy__2 = None
        copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_9, add_4);  primals_9 = add_4 = copy__3 = None
        copy__4: "f32[64]" = torch.ops.aten.copy_.default(primals_12, add_6);  primals_12 = add_6 = copy__4 = None
        copy__5: "f32[64]" = torch.ops.aten.copy_.default(primals_13, add_7);  primals_13 = add_7 = copy__5 = None
        copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_15, add_8);  primals_15 = add_8 = copy__6 = None
        copy__7: "f32[64]" = torch.ops.aten.copy_.default(primals_18, add_10);  primals_18 = add_10 = copy__7 = None
        copy__8: "f32[64]" = torch.ops.aten.copy_.default(primals_19, add_11);  primals_19 = add_11 = copy__8 = None
        copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_21, add_12);  primals_21 = add_12 = copy__9 = None
        copy__10: "f32[256]" = torch.ops.aten.copy_.default(primals_24, add_14);  primals_24 = add_14 = copy__10 = None
        copy__11: "f32[256]" = torch.ops.aten.copy_.default(primals_25, add_15);  primals_25 = add_15 = copy__11 = None
        copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_27, add_16);  primals_27 = add_16 = copy__12 = None
        copy__13: "f32[256]" = torch.ops.aten.copy_.default(primals_30, add_18);  primals_30 = add_18 = copy__13 = None
        copy__14: "f32[256]" = torch.ops.aten.copy_.default(primals_31, add_19);  primals_31 = add_19 = copy__14 = None
        copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_33, add_21);  primals_33 = add_21 = copy__15 = None
        copy__16: "f32[64]" = torch.ops.aten.copy_.default(primals_36, add_23);  primals_36 = add_23 = copy__16 = None
        copy__17: "f32[64]" = torch.ops.aten.copy_.default(primals_37, add_24);  primals_37 = add_24 = copy__17 = None
        copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_39, add_25);  primals_39 = add_25 = copy__18 = None
        copy__19: "f32[64]" = torch.ops.aten.copy_.default(primals_42, add_27);  primals_42 = add_27 = copy__19 = None
        copy__20: "f32[64]" = torch.ops.aten.copy_.default(primals_43, add_28);  primals_43 = add_28 = copy__20 = None
        copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_45, add_29);  primals_45 = add_29 = copy__21 = None
        copy__22: "f32[256]" = torch.ops.aten.copy_.default(primals_48, add_31);  primals_48 = add_31 = copy__22 = None
        copy__23: "f32[256]" = torch.ops.aten.copy_.default(primals_49, add_32);  primals_49 = add_32 = copy__23 = None
        copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_51, add_34);  primals_51 = add_34 = copy__24 = None
        copy__25: "f32[64]" = torch.ops.aten.copy_.default(primals_54, add_36);  primals_54 = add_36 = copy__25 = None
        copy__26: "f32[64]" = torch.ops.aten.copy_.default(primals_55, add_37);  primals_55 = add_37 = copy__26 = None
        copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_57, add_38);  primals_57 = add_38 = copy__27 = None
        copy__28: "f32[64]" = torch.ops.aten.copy_.default(primals_60, add_40);  primals_60 = add_40 = copy__28 = None
        copy__29: "f32[64]" = torch.ops.aten.copy_.default(primals_61, add_41);  primals_61 = add_41 = copy__29 = None
        copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_63, add_42);  primals_63 = add_42 = copy__30 = None
        copy__31: "f32[256]" = torch.ops.aten.copy_.default(primals_66, add_44);  primals_66 = add_44 = copy__31 = None
        copy__32: "f32[256]" = torch.ops.aten.copy_.default(primals_67, add_45);  primals_67 = add_45 = copy__32 = None
        copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_69, add_47);  primals_69 = add_47 = copy__33 = None
        copy__34: "f32[128]" = torch.ops.aten.copy_.default(primals_72, add_49);  primals_72 = add_49 = copy__34 = None
        copy__35: "f32[128]" = torch.ops.aten.copy_.default(primals_73, add_50);  primals_73 = add_50 = copy__35 = None
        copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_75, add_51);  primals_75 = add_51 = copy__36 = None
        copy__37: "f32[128]" = torch.ops.aten.copy_.default(primals_78, add_53);  primals_78 = add_53 = copy__37 = None
        copy__38: "f32[128]" = torch.ops.aten.copy_.default(primals_79, add_54);  primals_79 = add_54 = copy__38 = None
        copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_81, add_55);  primals_81 = add_55 = copy__39 = None
        copy__40: "f32[512]" = torch.ops.aten.copy_.default(primals_84, add_57);  primals_84 = add_57 = copy__40 = None
        copy__41: "f32[512]" = torch.ops.aten.copy_.default(primals_85, add_58);  primals_85 = add_58 = copy__41 = None
        copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_87, add_59);  primals_87 = add_59 = copy__42 = None
        copy__43: "f32[512]" = torch.ops.aten.copy_.default(primals_90, add_61);  primals_90 = add_61 = copy__43 = None
        copy__44: "f32[512]" = torch.ops.aten.copy_.default(primals_91, add_62);  primals_91 = add_62 = copy__44 = None
        copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_93, add_64);  primals_93 = add_64 = copy__45 = None
        copy__46: "f32[128]" = torch.ops.aten.copy_.default(primals_96, add_66);  primals_96 = add_66 = copy__46 = None
        copy__47: "f32[128]" = torch.ops.aten.copy_.default(primals_97, add_67);  primals_97 = add_67 = copy__47 = None
        copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_99, add_68);  primals_99 = add_68 = copy__48 = None
        copy__49: "f32[128]" = torch.ops.aten.copy_.default(primals_102, add_70);  primals_102 = add_70 = copy__49 = None
        copy__50: "f32[128]" = torch.ops.aten.copy_.default(primals_103, add_71);  primals_103 = add_71 = copy__50 = None
        copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_105, add_72);  primals_105 = add_72 = copy__51 = None
        copy__52: "f32[512]" = torch.ops.aten.copy_.default(primals_108, add_74);  primals_108 = add_74 = copy__52 = None
        copy__53: "f32[512]" = torch.ops.aten.copy_.default(primals_109, add_75);  primals_109 = add_75 = copy__53 = None
        copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_111, add_77);  primals_111 = add_77 = copy__54 = None
        copy__55: "f32[128]" = torch.ops.aten.copy_.default(primals_114, add_79);  primals_114 = add_79 = copy__55 = None
        copy__56: "f32[128]" = torch.ops.aten.copy_.default(primals_115, add_80);  primals_115 = add_80 = copy__56 = None
        copy__57: "i64[]" = torch.ops.aten.copy_.default(primals_117, add_81);  primals_117 = add_81 = copy__57 = None
        copy__58: "f32[128]" = torch.ops.aten.copy_.default(primals_120, add_83);  primals_120 = add_83 = copy__58 = None
        copy__59: "f32[128]" = torch.ops.aten.copy_.default(primals_121, add_84);  primals_121 = add_84 = copy__59 = None
        copy__60: "i64[]" = torch.ops.aten.copy_.default(primals_123, add_85);  primals_123 = add_85 = copy__60 = None
        copy__61: "f32[512]" = torch.ops.aten.copy_.default(primals_126, add_87);  primals_126 = add_87 = copy__61 = None
        copy__62: "f32[512]" = torch.ops.aten.copy_.default(primals_127, add_88);  primals_127 = add_88 = copy__62 = None
        copy__63: "i64[]" = torch.ops.aten.copy_.default(primals_129, add_90);  primals_129 = add_90 = copy__63 = None
        copy__64: "f32[128]" = torch.ops.aten.copy_.default(primals_132, add_92);  primals_132 = add_92 = copy__64 = None
        copy__65: "f32[128]" = torch.ops.aten.copy_.default(primals_133, add_93);  primals_133 = add_93 = copy__65 = None
        copy__66: "i64[]" = torch.ops.aten.copy_.default(primals_135, add_94);  primals_135 = add_94 = copy__66 = None
        copy__67: "f32[128]" = torch.ops.aten.copy_.default(primals_138, add_96);  primals_138 = add_96 = copy__67 = None
        copy__68: "f32[128]" = torch.ops.aten.copy_.default(primals_139, add_97);  primals_139 = add_97 = copy__68 = None
        copy__69: "i64[]" = torch.ops.aten.copy_.default(primals_141, add_98);  primals_141 = add_98 = copy__69 = None
        copy__70: "f32[512]" = torch.ops.aten.copy_.default(primals_144, add_100);  primals_144 = add_100 = copy__70 = None
        copy__71: "f32[512]" = torch.ops.aten.copy_.default(primals_145, add_101);  primals_145 = add_101 = copy__71 = None
        copy__72: "i64[]" = torch.ops.aten.copy_.default(primals_147, add_103);  primals_147 = add_103 = copy__72 = None
        copy__73: "f32[256]" = torch.ops.aten.copy_.default(primals_150, add_105);  primals_150 = add_105 = copy__73 = None
        copy__74: "f32[256]" = torch.ops.aten.copy_.default(primals_151, add_106);  primals_151 = add_106 = copy__74 = None
        copy__75: "i64[]" = torch.ops.aten.copy_.default(primals_153, add_107);  primals_153 = add_107 = copy__75 = None
        copy__76: "f32[256]" = torch.ops.aten.copy_.default(primals_156, add_109);  primals_156 = add_109 = copy__76 = None
        copy__77: "f32[256]" = torch.ops.aten.copy_.default(primals_157, add_110);  primals_157 = add_110 = copy__77 = None
        copy__78: "i64[]" = torch.ops.aten.copy_.default(primals_159, add_111);  primals_159 = add_111 = copy__78 = None
        copy__79: "f32[1024]" = torch.ops.aten.copy_.default(primals_162, add_113);  primals_162 = add_113 = copy__79 = None
        copy__80: "f32[1024]" = torch.ops.aten.copy_.default(primals_163, add_114);  primals_163 = add_114 = copy__80 = None
        copy__81: "i64[]" = torch.ops.aten.copy_.default(primals_165, add_115);  primals_165 = add_115 = copy__81 = None
        copy__82: "f32[1024]" = torch.ops.aten.copy_.default(primals_168, add_117);  primals_168 = add_117 = copy__82 = None
        copy__83: "f32[1024]" = torch.ops.aten.copy_.default(primals_169, add_118);  primals_169 = add_118 = copy__83 = None
        copy__84: "i64[]" = torch.ops.aten.copy_.default(primals_171, add_120);  primals_171 = add_120 = copy__84 = None
        copy__85: "f32[256]" = torch.ops.aten.copy_.default(primals_174, add_122);  primals_174 = add_122 = copy__85 = None
        copy__86: "f32[256]" = torch.ops.aten.copy_.default(primals_175, add_123);  primals_175 = add_123 = copy__86 = None
        copy__87: "i64[]" = torch.ops.aten.copy_.default(primals_177, add_124);  primals_177 = add_124 = copy__87 = None
        copy__88: "f32[256]" = torch.ops.aten.copy_.default(primals_180, add_126);  primals_180 = add_126 = copy__88 = None
        copy__89: "f32[256]" = torch.ops.aten.copy_.default(primals_181, add_127);  primals_181 = add_127 = copy__89 = None
        copy__90: "i64[]" = torch.ops.aten.copy_.default(primals_183, add_128);  primals_183 = add_128 = copy__90 = None
        copy__91: "f32[1024]" = torch.ops.aten.copy_.default(primals_186, add_130);  primals_186 = add_130 = copy__91 = None
        copy__92: "f32[1024]" = torch.ops.aten.copy_.default(primals_187, add_131);  primals_187 = add_131 = copy__92 = None
        copy__93: "i64[]" = torch.ops.aten.copy_.default(primals_189, add_133);  primals_189 = add_133 = copy__93 = None
        copy__94: "f32[256]" = torch.ops.aten.copy_.default(primals_192, add_135);  primals_192 = add_135 = copy__94 = None
        copy__95: "f32[256]" = torch.ops.aten.copy_.default(primals_193, add_136);  primals_193 = add_136 = copy__95 = None
        copy__96: "i64[]" = torch.ops.aten.copy_.default(primals_195, add_137);  primals_195 = add_137 = copy__96 = None
        copy__97: "f32[256]" = torch.ops.aten.copy_.default(primals_198, add_139);  primals_198 = add_139 = copy__97 = None
        copy__98: "f32[256]" = torch.ops.aten.copy_.default(primals_199, add_140);  primals_199 = add_140 = copy__98 = None
        copy__99: "i64[]" = torch.ops.aten.copy_.default(primals_201, add_141);  primals_201 = add_141 = copy__99 = None
        copy__100: "f32[1024]" = torch.ops.aten.copy_.default(primals_204, add_143);  primals_204 = add_143 = copy__100 = None
        copy__101: "f32[1024]" = torch.ops.aten.copy_.default(primals_205, add_144);  primals_205 = add_144 = copy__101 = None
        copy__102: "i64[]" = torch.ops.aten.copy_.default(primals_207, add_146);  primals_207 = add_146 = copy__102 = None
        copy__103: "f32[256]" = torch.ops.aten.copy_.default(primals_210, add_148);  primals_210 = add_148 = copy__103 = None
        copy__104: "f32[256]" = torch.ops.aten.copy_.default(primals_211, add_149);  primals_211 = add_149 = copy__104 = None
        copy__105: "i64[]" = torch.ops.aten.copy_.default(primals_213, add_150);  primals_213 = add_150 = copy__105 = None
        copy__106: "f32[256]" = torch.ops.aten.copy_.default(primals_216, add_152);  primals_216 = add_152 = copy__106 = None
        copy__107: "f32[256]" = torch.ops.aten.copy_.default(primals_217, add_153);  primals_217 = add_153 = copy__107 = None
        copy__108: "i64[]" = torch.ops.aten.copy_.default(primals_219, add_154);  primals_219 = add_154 = copy__108 = None
        copy__109: "f32[1024]" = torch.ops.aten.copy_.default(primals_222, add_156);  primals_222 = add_156 = copy__109 = None
        copy__110: "f32[1024]" = torch.ops.aten.copy_.default(primals_223, add_157);  primals_223 = add_157 = copy__110 = None
        copy__111: "i64[]" = torch.ops.aten.copy_.default(primals_225, add_159);  primals_225 = add_159 = copy__111 = None
        copy__112: "f32[256]" = torch.ops.aten.copy_.default(primals_228, add_161);  primals_228 = add_161 = copy__112 = None
        copy__113: "f32[256]" = torch.ops.aten.copy_.default(primals_229, add_162);  primals_229 = add_162 = copy__113 = None
        copy__114: "i64[]" = torch.ops.aten.copy_.default(primals_231, add_163);  primals_231 = add_163 = copy__114 = None
        copy__115: "f32[256]" = torch.ops.aten.copy_.default(primals_234, add_165);  primals_234 = add_165 = copy__115 = None
        copy__116: "f32[256]" = torch.ops.aten.copy_.default(primals_235, add_166);  primals_235 = add_166 = copy__116 = None
        copy__117: "i64[]" = torch.ops.aten.copy_.default(primals_237, add_167);  primals_237 = add_167 = copy__117 = None
        copy__118: "f32[1024]" = torch.ops.aten.copy_.default(primals_240, add_169);  primals_240 = add_169 = copy__118 = None
        copy__119: "f32[1024]" = torch.ops.aten.copy_.default(primals_241, add_170);  primals_241 = add_170 = copy__119 = None
        copy__120: "i64[]" = torch.ops.aten.copy_.default(primals_243, add_172);  primals_243 = add_172 = copy__120 = None
        copy__121: "f32[256]" = torch.ops.aten.copy_.default(primals_246, add_174);  primals_246 = add_174 = copy__121 = None
        copy__122: "f32[256]" = torch.ops.aten.copy_.default(primals_247, add_175);  primals_247 = add_175 = copy__122 = None
        copy__123: "i64[]" = torch.ops.aten.copy_.default(primals_249, add_176);  primals_249 = add_176 = copy__123 = None
        copy__124: "f32[256]" = torch.ops.aten.copy_.default(primals_252, add_178);  primals_252 = add_178 = copy__124 = None
        copy__125: "f32[256]" = torch.ops.aten.copy_.default(primals_253, add_179);  primals_253 = add_179 = copy__125 = None
        copy__126: "i64[]" = torch.ops.aten.copy_.default(primals_255, add_180);  primals_255 = add_180 = copy__126 = None
        copy__127: "f32[1024]" = torch.ops.aten.copy_.default(primals_258, add_182);  primals_258 = add_182 = copy__127 = None
        copy__128: "f32[1024]" = torch.ops.aten.copy_.default(primals_259, add_183);  primals_259 = add_183 = copy__128 = None
        copy__129: "i64[]" = torch.ops.aten.copy_.default(primals_261, add_185);  primals_261 = add_185 = copy__129 = None
        copy__130: "f32[512]" = torch.ops.aten.copy_.default(primals_264, add_187);  primals_264 = add_187 = copy__130 = None
        copy__131: "f32[512]" = torch.ops.aten.copy_.default(primals_265, add_188);  primals_265 = add_188 = copy__131 = None
        copy__132: "i64[]" = torch.ops.aten.copy_.default(primals_267, add_189);  primals_267 = add_189 = copy__132 = None
        copy__133: "f32[512]" = torch.ops.aten.copy_.default(primals_270, add_191);  primals_270 = add_191 = copy__133 = None
        copy__134: "f32[512]" = torch.ops.aten.copy_.default(primals_271, add_192);  primals_271 = add_192 = copy__134 = None
        copy__135: "i64[]" = torch.ops.aten.copy_.default(primals_273, add_193);  primals_273 = add_193 = copy__135 = None
        copy__136: "f32[2048]" = torch.ops.aten.copy_.default(primals_276, add_195);  primals_276 = add_195 = copy__136 = None
        copy__137: "f32[2048]" = torch.ops.aten.copy_.default(primals_277, add_196);  primals_277 = add_196 = copy__137 = None
        copy__138: "i64[]" = torch.ops.aten.copy_.default(primals_279, add_197);  primals_279 = add_197 = copy__138 = None
        copy__139: "f32[2048]" = torch.ops.aten.copy_.default(primals_282, add_199);  primals_282 = add_199 = copy__139 = None
        copy__140: "f32[2048]" = torch.ops.aten.copy_.default(primals_283, add_200);  primals_283 = add_200 = copy__140 = None
        copy__141: "i64[]" = torch.ops.aten.copy_.default(primals_285, add_202);  primals_285 = add_202 = copy__141 = None
        copy__142: "f32[512]" = torch.ops.aten.copy_.default(primals_288, add_204);  primals_288 = add_204 = copy__142 = None
        copy__143: "f32[512]" = torch.ops.aten.copy_.default(primals_289, add_205);  primals_289 = add_205 = copy__143 = None
        copy__144: "i64[]" = torch.ops.aten.copy_.default(primals_291, add_206);  primals_291 = add_206 = copy__144 = None
        copy__145: "f32[512]" = torch.ops.aten.copy_.default(primals_294, add_208);  primals_294 = add_208 = copy__145 = None
        copy__146: "f32[512]" = torch.ops.aten.copy_.default(primals_295, add_209);  primals_295 = add_209 = copy__146 = None
        copy__147: "i64[]" = torch.ops.aten.copy_.default(primals_297, add_210);  primals_297 = add_210 = copy__147 = None
        copy__148: "f32[2048]" = torch.ops.aten.copy_.default(primals_300, add_212);  primals_300 = add_212 = copy__148 = None
        copy__149: "f32[2048]" = torch.ops.aten.copy_.default(primals_301, add_213);  primals_301 = add_213 = copy__149 = None
        copy__150: "i64[]" = torch.ops.aten.copy_.default(primals_303, add_215);  primals_303 = add_215 = copy__150 = None
        copy__151: "f32[512]" = torch.ops.aten.copy_.default(primals_306, add_217);  primals_306 = add_217 = copy__151 = None
        copy__152: "f32[512]" = torch.ops.aten.copy_.default(primals_307, add_218);  primals_307 = add_218 = copy__152 = None
        copy__153: "i64[]" = torch.ops.aten.copy_.default(primals_309, add_219);  primals_309 = add_219 = copy__153 = None
        copy__154: "f32[512]" = torch.ops.aten.copy_.default(primals_312, add_221);  primals_312 = add_221 = copy__154 = None
        copy__155: "f32[512]" = torch.ops.aten.copy_.default(primals_313, add_222);  primals_313 = add_222 = copy__155 = None
        copy__156: "i64[]" = torch.ops.aten.copy_.default(primals_315, add_223);  primals_315 = add_223 = copy__156 = None
        copy__157: "f32[2048]" = torch.ops.aten.copy_.default(primals_318, add_225);  primals_318 = add_225 = copy__157 = None
        copy__158: "f32[2048]" = torch.ops.aten.copy_.default(primals_319, add_226);  primals_319 = add_226 = copy__158 = None
        return (mm, primals_4, primals_10, primals_16, primals_22, primals_28, primals_34, primals_40, primals_46, primals_52, primals_58, primals_64, primals_70, primals_76, primals_82, primals_88, primals_94, primals_100, primals_106, primals_112, primals_118, primals_124, primals_130, primals_136, primals_142, primals_148, primals_154, primals_160, primals_166, primals_172, primals_178, primals_184, primals_190, primals_196, primals_202, primals_208, primals_214, primals_220, primals_226, primals_232, primals_238, primals_244, primals_250, primals_256, primals_262, primals_268, primals_274, primals_280, primals_286, primals_292, primals_298, primals_304, primals_310, primals_316, getitem, getitem_1, getitem_2, rsqrt, getitem_7, getitem_8, getitem_9, getitem_10, getitem_12, getitem_14, convert_element_type_2, getitem_15, getitem_16, getitem_17, rsqrt_1, getitem_22, getitem_23, getitem_24, getitem_27, convert_element_type_3, getitem_28, getitem_29, getitem_30, rsqrt_2, getitem_35, getitem_36, getitem_37, getitem_40, convert_element_type_4, getitem_41, getitem_42, getitem_43, rsqrt_3, getitem_48, getitem_49, getitem_50, convert_element_type_5, getitem_51, getitem_52, getitem_53, rsqrt_4, getitem_58, getitem_59, getitem_60, getitem_63, convert_element_type_6, getitem_64, getitem_65, getitem_66, rsqrt_5, getitem_71, getitem_72, getitem_73, getitem_76, convert_element_type_7, getitem_77, getitem_78, getitem_79, rsqrt_6, getitem_84, getitem_85, getitem_86, getitem_89, convert_element_type_8, getitem_90, getitem_91, getitem_92, rsqrt_7, getitem_97, getitem_98, getitem_99, getitem_102, convert_element_type_9, getitem_103, getitem_104, getitem_105, rsqrt_8, getitem_110, getitem_111, getitem_112, getitem_115, convert_element_type_10, getitem_116, getitem_117, getitem_118, rsqrt_9, getitem_123, getitem_124, getitem_125, getitem_128, convert_element_type_11, getitem_129, getitem_130, getitem_131, rsqrt_10, getitem_136, getitem_137, getitem_138, getitem_141, convert_element_type_12, getitem_142, getitem_143, getitem_144, rsqrt_11, getitem_149, getitem_150, getitem_151, getitem_154, convert_element_type_13, getitem_155, getitem_156, getitem_157, rsqrt_12, getitem_162, getitem_163, getitem_164, getitem_167, convert_element_type_14, getitem_168, getitem_169, getitem_170, rsqrt_13, getitem_175, getitem_176, getitem_177, convert_element_type_15, getitem_178, getitem_179, getitem_180, rsqrt_14, getitem_185, getitem_186, getitem_187, getitem_190, convert_element_type_16, getitem_191, getitem_192, getitem_193, rsqrt_15, getitem_198, getitem_199, getitem_200, getitem_203, convert_element_type_17, getitem_204, getitem_205, getitem_206, rsqrt_16, getitem_211, getitem_212, getitem_213, getitem_216, convert_element_type_18, getitem_217, getitem_218, getitem_219, rsqrt_17, getitem_224, getitem_225, getitem_226, getitem_229, convert_element_type_19, getitem_230, getitem_231, getitem_232, rsqrt_18, getitem_237, getitem_238, getitem_239, getitem_242, convert_element_type_20, getitem_243, getitem_244, getitem_245, rsqrt_19, getitem_250, getitem_251, getitem_252, getitem_255, convert_element_type_21, getitem_256, getitem_257, getitem_258, rsqrt_20, getitem_263, getitem_264, getitem_265, getitem_268, convert_element_type_22, getitem_269, getitem_270, getitem_271, rsqrt_21, getitem_276, getitem_277, getitem_278, getitem_281, convert_element_type_23, getitem_282, getitem_283, getitem_284, rsqrt_22, getitem_289, getitem_290, getitem_291, getitem_294, convert_element_type_24, getitem_295, getitem_296, getitem_297, rsqrt_23, getitem_302, getitem_303, getitem_304, getitem_307, convert_element_type_25, getitem_308, getitem_309, getitem_310, rsqrt_24, getitem_315, getitem_316, getitem_317, getitem_320, convert_element_type_26, getitem_321, getitem_322, getitem_323, rsqrt_25, getitem_328, getitem_329, getitem_330, getitem_333, convert_element_type_27, getitem_334, getitem_335, getitem_336, rsqrt_26, getitem_341, getitem_342, getitem_343, convert_element_type_28, getitem_344, getitem_345, getitem_346, rsqrt_27, getitem_351, getitem_352, getitem_353, getitem_356, convert_element_type_29, getitem_357, getitem_358, getitem_359, rsqrt_28, getitem_364, getitem_365, getitem_366, getitem_369, convert_element_type_30, getitem_370, getitem_371, getitem_372, rsqrt_29, getitem_377, getitem_378, getitem_379, getitem_382, convert_element_type_31, getitem_383, getitem_384, getitem_385, rsqrt_30, getitem_390, getitem_391, getitem_392, getitem_395, convert_element_type_32, getitem_396, getitem_397, getitem_398, rsqrt_31, getitem_403, getitem_404, getitem_405, getitem_408, convert_element_type_33, getitem_409, getitem_410, getitem_411, rsqrt_32, getitem_416, getitem_417, getitem_418, getitem_421, convert_element_type_34, getitem_422, getitem_423, getitem_424, rsqrt_33, getitem_429, getitem_430, getitem_431, getitem_434, convert_element_type_35, getitem_435, getitem_436, getitem_437, rsqrt_34, getitem_442, getitem_443, getitem_444, getitem_447, convert_element_type_36, getitem_448, getitem_449, getitem_450, rsqrt_35, getitem_455, getitem_456, getitem_457, getitem_460, convert_element_type_37, getitem_461, getitem_462, getitem_463, rsqrt_36, getitem_468, getitem_469, getitem_470, getitem_473, convert_element_type_38, getitem_474, getitem_475, getitem_476, rsqrt_37, getitem_481, getitem_482, getitem_483, getitem_486, convert_element_type_39, getitem_487, getitem_488, getitem_489, rsqrt_38, getitem_494, getitem_495, getitem_496, getitem_499, convert_element_type_40, getitem_500, getitem_501, getitem_502, rsqrt_39, getitem_507, getitem_508, getitem_509, getitem_512, convert_element_type_41, getitem_513, getitem_514, getitem_515, rsqrt_40, getitem_520, getitem_521, getitem_522, getitem_525, convert_element_type_42, getitem_526, getitem_527, getitem_528, rsqrt_41, getitem_533, getitem_534, getitem_535, getitem_538, convert_element_type_43, getitem_539, getitem_540, getitem_541, rsqrt_42, getitem_546, getitem_547, getitem_548, getitem_551, convert_element_type_44, getitem_552, getitem_553, getitem_554, rsqrt_43, getitem_559, getitem_560, getitem_561, getitem_564, convert_element_type_45, getitem_565, getitem_566, getitem_567, rsqrt_44, getitem_572, getitem_573, getitem_574, getitem_577, convert_element_type_46, getitem_578, getitem_579, getitem_580, rsqrt_45, getitem_585, getitem_586, getitem_587, convert_element_type_47, getitem_588, getitem_589, getitem_590, rsqrt_46, getitem_595, getitem_596, getitem_597, getitem_600, convert_element_type_48, getitem_601, getitem_602, getitem_603, rsqrt_47, getitem_608, getitem_609, getitem_610, getitem_613, convert_element_type_49, getitem_614, getitem_615, getitem_616, rsqrt_48, getitem_621, getitem_622, getitem_623, getitem_626, convert_element_type_50, getitem_627, getitem_628, getitem_629, rsqrt_49, getitem_634, getitem_635, getitem_636, getitem_639, convert_element_type_51, getitem_640, getitem_641, getitem_642, rsqrt_50, getitem_647, getitem_648, getitem_649, getitem_652, convert_element_type_52, getitem_653, getitem_654, getitem_655, rsqrt_51, getitem_660, getitem_661, getitem_662, getitem_665, convert_element_type_53, getitem_666, getitem_667, getitem_668, rsqrt_52, getitem_673, getitem_674, getitem_675, getitem_678, getitem_679, getitem_680, getitem_681, convert_element_type_54)
        