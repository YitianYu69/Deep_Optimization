class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 3, 7, 7]", primals_2: "f32[512, 3, 224, 224]", primals_3: "i64[]", primals_4: "f32[64]", primals_5: "f32[64]", primals_6: "f32[64]", primals_7: "f32[64]", primals_8: "f32[64, 64, 1, 1]", primals_9: "i64[]", primals_10: "f32[64]", primals_11: "f32[64]", primals_12: "f32[64]", primals_13: "f32[64]", primals_14: "f32[64, 64, 3, 3]", primals_15: "i64[]", primals_16: "f32[64]", primals_17: "f32[64]", primals_18: "f32[64]", primals_19: "f32[64]", primals_20: "f32[256, 64, 1, 1]", primals_21: "i64[]", primals_22: "f32[256]", primals_23: "f32[256]", primals_24: "f32[256]", primals_25: "f32[256]", primals_26: "f32[256, 64, 1, 1]", primals_27: "i64[]", primals_28: "f32[256]", primals_29: "f32[256]", primals_30: "f32[256]", primals_31: "f32[256]", primals_32: "f32[64, 256, 1, 1]", primals_33: "i64[]", primals_34: "f32[64]", primals_35: "f32[64]", primals_36: "f32[64]", primals_37: "f32[64]", primals_38: "f32[64, 64, 3, 3]", primals_39: "i64[]", primals_40: "f32[64]", primals_41: "f32[64]", primals_42: "f32[64]", primals_43: "f32[64]", primals_44: "f32[256, 64, 1, 1]", primals_45: "i64[]", primals_46: "f32[256]", primals_47: "f32[256]", primals_48: "f32[256]", primals_49: "f32[256]", primals_50: "f32[64, 256, 1, 1]", primals_51: "i64[]", primals_52: "f32[64]", primals_53: "f32[64]", primals_54: "f32[64]", primals_55: "f32[64]", primals_56: "f32[64, 64, 3, 3]", primals_57: "i64[]", primals_58: "f32[64]", primals_59: "f32[64]", primals_60: "f32[64]", primals_61: "f32[64]", primals_62: "f32[256, 64, 1, 1]", primals_63: "i64[]", primals_64: "f32[256]", primals_65: "f32[256]", primals_66: "f32[256]", primals_67: "f32[256]", primals_68: "f32[128, 256, 1, 1]", primals_69: "i64[]", primals_70: "f32[128]", primals_71: "f32[128]", primals_72: "f32[128]", primals_73: "f32[128]", primals_74: "f32[128, 128, 3, 3]", primals_75: "i64[]", primals_76: "f32[128]", primals_77: "f32[128]", primals_78: "f32[128]", primals_79: "f32[128]", primals_80: "f32[512, 128, 1, 1]", primals_81: "i64[]", primals_82: "f32[512]", primals_83: "f32[512]", primals_84: "f32[512]", primals_85: "f32[512]", primals_86: "f32[512, 256, 1, 1]", primals_87: "i64[]", primals_88: "f32[512]", primals_89: "f32[512]", primals_90: "f32[512]", primals_91: "f32[512]", primals_92: "f32[128, 512, 1, 1]", primals_93: "i64[]", primals_94: "f32[128]", primals_95: "f32[128]", primals_96: "f32[128]", primals_97: "f32[128]", primals_98: "f32[128, 128, 3, 3]", primals_99: "i64[]", primals_100: "f32[128]", primals_101: "f32[128]", primals_102: "f32[128]", primals_103: "f32[128]", primals_104: "f32[512, 128, 1, 1]", primals_105: "i64[]", primals_106: "f32[512]", primals_107: "f32[512]", primals_108: "f32[512]", primals_109: "f32[512]", primals_110: "f32[128, 512, 1, 1]", primals_111: "i64[]", primals_112: "f32[128]", primals_113: "f32[128]", primals_114: "f32[128]", primals_115: "f32[128]", primals_116: "f32[128, 128, 3, 3]", primals_117: "i64[]", primals_118: "f32[128]", primals_119: "f32[128]", primals_120: "f32[128]", primals_121: "f32[128]", primals_122: "f32[512, 128, 1, 1]", primals_123: "i64[]", primals_124: "f32[512]", primals_125: "f32[512]", primals_126: "f32[512]", primals_127: "f32[512]", primals_128: "f32[128, 512, 1, 1]", primals_129: "i64[]", primals_130: "f32[128]", primals_131: "f32[128]", primals_132: "f32[128]", primals_133: "f32[128]", primals_134: "f32[128, 128, 3, 3]", primals_135: "i64[]", primals_136: "f32[128]", primals_137: "f32[128]", primals_138: "f32[128]", primals_139: "f32[128]", primals_140: "f32[512, 128, 1, 1]", primals_141: "i64[]", primals_142: "f32[512]", primals_143: "f32[512]", primals_144: "f32[512]", primals_145: "f32[512]", primals_146: "f32[256, 512, 1, 1]", primals_147: "i64[]", primals_148: "f32[256]", primals_149: "f32[256]", primals_150: "f32[256]", primals_151: "f32[256]", primals_152: "f32[256, 256, 3, 3]", primals_153: "i64[]", primals_154: "f32[256]", primals_155: "f32[256]", primals_156: "f32[256]", primals_157: "f32[256]", primals_158: "f32[1024, 256, 1, 1]", primals_159: "i64[]", primals_160: "f32[1024]", primals_161: "f32[1024]", primals_162: "f32[1024]", primals_163: "f32[1024]", primals_164: "f32[1024, 512, 1, 1]", primals_165: "i64[]", primals_166: "f32[1024]", primals_167: "f32[1024]", primals_168: "f32[1024]", primals_169: "f32[1024]", primals_170: "f32[256, 1024, 1, 1]", primals_171: "i64[]", primals_172: "f32[256]", primals_173: "f32[256]", primals_174: "f32[256]", primals_175: "f32[256]", primals_176: "f32[256, 256, 3, 3]", primals_177: "i64[]", primals_178: "f32[256]", primals_179: "f32[256]", primals_180: "f32[256]", primals_181: "f32[256]", primals_182: "f32[1024, 256, 1, 1]", primals_183: "i64[]", primals_184: "f32[1024]", primals_185: "f32[1024]", primals_186: "f32[1024]", primals_187: "f32[1024]", primals_188: "f32[256, 1024, 1, 1]", primals_189: "i64[]", primals_190: "f32[256]", primals_191: "f32[256]", primals_192: "f32[256]", primals_193: "f32[256]", primals_194: "f32[256, 256, 3, 3]", primals_195: "i64[]", primals_196: "f32[256]", primals_197: "f32[256]", primals_198: "f32[256]", primals_199: "f32[256]", primals_200: "f32[1024, 256, 1, 1]", primals_201: "i64[]", primals_202: "f32[1024]", primals_203: "f32[1024]", primals_204: "f32[1024]", primals_205: "f32[1024]", primals_206: "f32[256, 1024, 1, 1]", primals_207: "i64[]", primals_208: "f32[256]", primals_209: "f32[256]", primals_210: "f32[256]", primals_211: "f32[256]", primals_212: "f32[256, 256, 3, 3]", primals_213: "i64[]", primals_214: "f32[256]", primals_215: "f32[256]", primals_216: "f32[256]", primals_217: "f32[256]", primals_218: "f32[1024, 256, 1, 1]", primals_219: "i64[]", primals_220: "f32[1024]", primals_221: "f32[1024]", primals_222: "f32[1024]", primals_223: "f32[1024]", primals_224: "f32[256, 1024, 1, 1]", primals_225: "i64[]", primals_226: "f32[256]", primals_227: "f32[256]", primals_228: "f32[256]", primals_229: "f32[256]", primals_230: "f32[256, 256, 3, 3]", primals_231: "i64[]", primals_232: "f32[256]", primals_233: "f32[256]", primals_234: "f32[256]", primals_235: "f32[256]", primals_236: "f32[1024, 256, 1, 1]", primals_237: "i64[]", primals_238: "f32[1024]", primals_239: "f32[1024]", primals_240: "f32[1024]", primals_241: "f32[1024]", primals_242: "f32[256, 1024, 1, 1]", primals_243: "i64[]", primals_244: "f32[256]", primals_245: "f32[256]", primals_246: "f32[256]", primals_247: "f32[256]", primals_248: "f32[256, 256, 3, 3]", primals_249: "i64[]", primals_250: "f32[256]", primals_251: "f32[256]", primals_252: "f32[256]", primals_253: "f32[256]", primals_254: "f32[1024, 256, 1, 1]", primals_255: "i64[]", primals_256: "f32[1024]", primals_257: "f32[1024]", primals_258: "f32[1024]", primals_259: "f32[1024]", primals_260: "f32[512, 1024, 1, 1]", primals_261: "i64[]", primals_262: "f32[512]", primals_263: "f32[512]", primals_264: "f32[512]", primals_265: "f32[512]", primals_266: "f32[512, 512, 3, 3]", primals_267: "i64[]", primals_268: "f32[512]", primals_269: "f32[512]", primals_270: "f32[512]", primals_271: "f32[512]", primals_272: "f32[2048, 512, 1, 1]", primals_273: "i64[]", primals_274: "f32[2048]", primals_275: "f32[2048]", primals_276: "f32[2048]", primals_277: "f32[2048]", primals_278: "f32[2048, 1024, 1, 1]", primals_279: "i64[]", primals_280: "f32[2048]", primals_281: "f32[2048]", primals_282: "f32[2048]", primals_283: "f32[2048]", primals_284: "f32[512, 2048, 1, 1]", primals_285: "i64[]", primals_286: "f32[512]", primals_287: "f32[512]", primals_288: "f32[512]", primals_289: "f32[512]", primals_290: "f32[512, 512, 3, 3]", primals_291: "i64[]", primals_292: "f32[512]", primals_293: "f32[512]", primals_294: "f32[512]", primals_295: "f32[512]", primals_296: "f32[2048, 512, 1, 1]", primals_297: "i64[]", primals_298: "f32[2048]", primals_299: "f32[2048]", primals_300: "f32[2048]", primals_301: "f32[2048]", primals_302: "f32[512, 2048, 1, 1]", primals_303: "i64[]", primals_304: "f32[512]", primals_305: "f32[512]", primals_306: "f32[512]", primals_307: "f32[512]", primals_308: "f32[512, 512, 3, 3]", primals_309: "i64[]", primals_310: "f32[512]", primals_311: "f32[512]", primals_312: "f32[512]", primals_313: "f32[512]", primals_314: "f32[2048, 512, 1, 1]", primals_315: "i64[]", primals_316: "f32[2048]", primals_317: "f32[2048]", primals_318: "f32[2048]", primals_319: "f32[2048]", primals_320: "f32[100, 2048]"):
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type: "bf16[512, 3, 224, 224]" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convert_element_type_1: "bf16[64, 3, 7, 7]" = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16);  primals_1 = None
        view: "bf16[512, 294, 512]" = torch.ops.aten.reshape.default(convert_element_type, [512, -1, 512])
        view_1: "bf16[150528, 512]" = torch.ops.aten.reshape.default(view, [150528, 512]);  view = None
        empty: "i32[150528, 32]" = torch.ops.aten.empty.memory_format([150528, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_1: "bf16[150528]" = torch.ops.aten.empty.memory_format([150528], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_2: "bf16[150528]" = torch.ops.aten.empty.memory_format([150528], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_310 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 218, constant_args_idx = 311, grid = [(150528, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1, 'P_ptr': empty, 'S_ptr': empty_1, 'M_ptr': empty_2, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_1 = triton_kernel_wrapper_mutation_310 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution: "bf16[512, 64, 112, 112]" = torch.ops.aten.convolution.default(convert_element_type, convert_element_type_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1);  convert_element_type = convert_element_type_1 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add: "i64[]" = torch.ops.aten.add.Tensor(primals_3, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_7: "bf16[512, 64, 12544]" = torch.ops.aten.reshape.default(convolution, [512, 64, 12544]);  convolution = None
        full_default: "f32[64]" = torch.ops.aten.full.default([64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        as_strided_default_356: "f32[64]" = torch.ops.aten.as_strided.default(full_default, [64], [1], 0)
        clone_default_178: "f32[64]" = torch.ops.aten.clone.default(as_strided_default_356);  as_strided_default_356 = None
        as_strided_default_357: "f32[64]" = torch.ops.aten.as_strided.default(clone_default_178, [64], [1], 0);  clone_default_178 = None
        as_strided_default_358: "f32[64]" = torch.ops.aten.as_strided.default(full_default, [64], [1], 0)
        clone_default_179: "f32[64]" = torch.ops.aten.clone.default(as_strided_default_358);  as_strided_default_358 = None
        as_strided_default_359: "f32[64]" = torch.ops.aten.as_strided.default(clone_default_179, [64], [1], 0);  clone_default_179 = None
        triton_kernel_wrapper_mutation_309 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 219, constant_args_idx = 312, grid = [(64, 6272, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_7, 'SUM': as_strided_default_357, 'SUMSQ': as_strided_default_359, 'M': 6422528, 'HW': 12544, 'stride_n': 802816, 'stride_c': 12544, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_309 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 6422528.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        div: "f32[64]" = torch.ops.aten.div.Tensor(as_strided_default_357, full_default_2);  as_strided_default_357 = None
        div_1: "f32[64]" = torch.ops.aten.div.Tensor(as_strided_default_359, full_default_2);  as_strided_default_359 = full_default_2 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_308 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 220, constant_args_idx = 313, grid = [(64, 6272, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_7, 'MEAN': div, 'INVSTD': rsqrt, 'GAMMA': primals_4, 'BETA': primals_5, 'Y': permute, 'X_hat': permute_1, 'M': 6422528, 'HW': 12544, 'stride_n': 802816, 'stride_c': 12544, 'BLOCK_M': 1024});  view_7 = div = primals_5 = triton_kernel_wrapper_mutation_308 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_5: "i32[802816, 32]" = torch.ops.aten.empty.memory_format([802816, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_6: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_7: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_11: "bf16[512, 1568, 512]" = torch.ops.aten.reshape.default(permute_1, [512, -1, 512]);  permute_1 = None
        view_12: "bf16[802816, 512]" = torch.ops.aten.reshape.default(view_11, [802816, 512]);  view_11 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_307 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 221, constant_args_idx = 314, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_12, 'P_ptr': empty_5, 'S_ptr': empty_6, 'M_ptr': empty_7, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_12 = triton_kernel_wrapper_mutation_307 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_4: "i8[512, 64, 112, 112]" = torch.ops.aten.full.default([512, 64, 112, 112], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_18: "bf16[512, 64, 112, 112]" = torch.ops.aten.reshape.default(permute, [512, 64, 112, 112]);  permute = None
        empty_8: "bf16[512, 64, 112, 112]" = torch.ops.aten.empty.memory_format([512, 64, 112, 112], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_2: "bf16[512, 64, 112, 112]" = torch.ops.aten.permute.default(empty_8, [0, 1, 2, 3]);  empty_8 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_306 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 315, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_18, 'Y_ptr': permute_2, 'Mask_prt': full_default_4, 'n_elts': 411041792, 'BLOCK_SIZE': 1024});  view_18 = triton_kernel_wrapper_mutation_306 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_5: "i32[802816, 16]" = torch.ops.aten.full.default([802816, 16], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_21: "i8[512, 1568, 512]" = torch.ops.aten.reshape.default(full_default_4, [512, -1, 512]);  full_default_4 = None
        view_22: "i8[802816, 512]" = torch.ops.aten.reshape.default(view_21, [802816, 512]);  view_21 = None
        
        # No stacktrace found for following nodes
        as_strided_default_354: "i32[12845056]" = torch.ops.aten.as_strided.default(full_default_5, [12845056], [1], 0)
        clone_default_177: "i32[12845056]" = torch.ops.aten.clone.default(as_strided_default_354);  as_strided_default_354 = None
        as_strided_default_355: "i32[802816, 16]" = torch.ops.aten.as_strided.default(clone_default_177, [802816, 16], [16, 1], 0);  clone_default_177 = None
        triton_kernel_wrapper_mutation_305 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 316, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_22, 'P_ptr': as_strided_default_355, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_22 = triton_kernel_wrapper_mutation_305 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:8 in forward, code: maxpool = self.maxpool(relu);  relu = None
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(permute_2, [3, 3], [2, 2], [1, 1], [1, 1], False)
        getitem_13: "bf16[512, 64, 56, 56]" = _low_memory_max_pool_with_offsets[0]
        getitem_14: "i8[512, 64, 56, 56]" = _low_memory_max_pool_with_offsets[1];  _low_memory_max_pool_with_offsets = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_2: "bf16[64, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        view_23: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(getitem_13, [512, -1, 512])
        view_24: "bf16[200704, 512]" = torch.ops.aten.reshape.default(view_23, [200704, 512]);  view_23 = None
        empty_9: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_10: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_11: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_304 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 222, constant_args_idx = 317, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_24, 'P_ptr': empty_9, 'S_ptr': empty_10, 'M_ptr': empty_11, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  triton_kernel_wrapper_mutation_304 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_1: "bf16[512, 64, 56, 56]" = torch.ops.aten.convolution.default(getitem_13, convert_element_type_2, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_4: "i64[]" = torch.ops.aten.add.Tensor(primals_9, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_30: "bf16[512, 64, 3136]" = torch.ops.aten.reshape.default(convolution_1, [512, 64, 3136]);  convolution_1 = None
        
        # No stacktrace found for following nodes
        as_strided_default_350: "f32[64]" = torch.ops.aten.as_strided.default(full_default, [64], [1], 0)
        clone_default_175: "f32[64]" = torch.ops.aten.clone.default(as_strided_default_350);  as_strided_default_350 = None
        as_strided_default_351: "f32[64]" = torch.ops.aten.as_strided.default(clone_default_175, [64], [1], 0);  clone_default_175 = None
        as_strided_default_352: "f32[64]" = torch.ops.aten.as_strided.default(full_default, [64], [1], 0)
        clone_default_176: "f32[64]" = torch.ops.aten.clone.default(as_strided_default_352);  as_strided_default_352 = None
        as_strided_default_353: "f32[64]" = torch.ops.aten.as_strided.default(clone_default_176, [64], [1], 0);  clone_default_176 = None
        triton_kernel_wrapper_mutation_303 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 223, constant_args_idx = 318, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_30, 'SUM': as_strided_default_351, 'SUMSQ': as_strided_default_353, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_303 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        full_default_8: "f32[]" = torch.ops.aten.full.default([], 1605632.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        div_3: "f32[64]" = torch.ops.aten.div.Tensor(as_strided_default_351, full_default_8);  as_strided_default_351 = None
        div_4: "f32[64]" = torch.ops.aten.div.Tensor(as_strided_default_353, full_default_8);  as_strided_default_353 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_302 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 224, constant_args_idx = 319, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_30, 'MEAN': div_3, 'INVSTD': rsqrt_1, 'GAMMA': primals_10, 'BETA': primals_11, 'Y': permute_3, 'X_hat': permute_4, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024});  view_30 = div_3 = primals_11 = triton_kernel_wrapper_mutation_302 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_14: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_15: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_16: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_34: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(permute_4, [512, -1, 512]);  permute_4 = None
        view_35: "bf16[200704, 512]" = torch.ops.aten.reshape.default(view_34, [200704, 512]);  view_34 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_301 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 225, constant_args_idx = 320, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_35, 'P_ptr': empty_14, 'S_ptr': empty_15, 'M_ptr': empty_16, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_35 = triton_kernel_wrapper_mutation_301 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_10: "i8[512, 64, 56, 56]" = torch.ops.aten.full.default([512, 64, 56, 56], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_41: "bf16[512, 64, 56, 56]" = torch.ops.aten.reshape.default(permute_3, [512, 64, 56, 56]);  permute_3 = None
        empty_17: "bf16[512, 64, 56, 56]" = torch.ops.aten.empty.memory_format([512, 64, 56, 56], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_5: "bf16[512, 64, 56, 56]" = torch.ops.aten.permute.default(empty_17, [0, 1, 2, 3]);  empty_17 = None
        
        # No stacktrace found for following nodes
        as_strided_default_348: "i8[102760448]" = torch.ops.aten.as_strided.default(full_default_10, [102760448], [1], 0)
        clone_default_174: "i8[102760448]" = torch.ops.aten.clone.default(as_strided_default_348);  as_strided_default_348 = None
        as_strided_default_349: "i8[512, 64, 56, 56]" = torch.ops.aten.as_strided.default(clone_default_174, [512, 64, 56, 56], [200704, 3136, 56, 1], 0);  clone_default_174 = None
        triton_kernel_wrapper_mutation_300 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 321, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_41, 'Y_ptr': permute_5, 'Mask_prt': as_strided_default_349, 'n_elts': 102760448, 'BLOCK_SIZE': 1024});  view_41 = triton_kernel_wrapper_mutation_300 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_11: "i32[200704, 16]" = torch.ops.aten.full.default([200704, 16], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_44: "i8[512, 392, 512]" = torch.ops.aten.reshape.default(as_strided_default_349, [512, -1, 512]);  as_strided_default_349 = None
        view_45: "i8[200704, 512]" = torch.ops.aten.reshape.default(view_44, [200704, 512]);  view_44 = None
        
        # No stacktrace found for following nodes
        as_strided_default_346: "i32[3211264]" = torch.ops.aten.as_strided.default(full_default_11, [3211264], [1], 0)
        clone_default_173: "i32[3211264]" = torch.ops.aten.clone.default(as_strided_default_346);  as_strided_default_346 = None
        as_strided_default_347: "i32[200704, 16]" = torch.ops.aten.as_strided.default(clone_default_173, [200704, 16], [16, 1], 0);  clone_default_173 = None
        triton_kernel_wrapper_mutation_299 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 322, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_45, 'P_ptr': as_strided_default_347, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_45 = triton_kernel_wrapper_mutation_299 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_3: "bf16[64, 64, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        empty_18: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_19: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_20: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_48: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(permute_5, [512, -1, 512])
        view_49: "bf16[200704, 512]" = torch.ops.aten.reshape.default(view_48, [200704, 512]);  view_48 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_298 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 226, constant_args_idx = 323, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_49, 'P_ptr': empty_18, 'S_ptr': empty_19, 'M_ptr': empty_20, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_49 = triton_kernel_wrapper_mutation_298 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_2: "bf16[512, 64, 56, 56]" = torch.ops.aten.convolution.default(permute_5, convert_element_type_3, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  permute_5 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_8: "i64[]" = torch.ops.aten.add.Tensor(primals_15, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_55: "bf16[512, 64, 3136]" = torch.ops.aten.reshape.default(convolution_2, [512, 64, 3136]);  convolution_2 = None
        
        # No stacktrace found for following nodes
        as_strided_default_342: "f32[64]" = torch.ops.aten.as_strided.default(full_default, [64], [1], 0)
        clone_default_171: "f32[64]" = torch.ops.aten.clone.default(as_strided_default_342);  as_strided_default_342 = None
        as_strided_default_343: "f32[64]" = torch.ops.aten.as_strided.default(clone_default_171, [64], [1], 0);  clone_default_171 = None
        as_strided_default_344: "f32[64]" = torch.ops.aten.as_strided.default(full_default, [64], [1], 0)
        clone_default_172: "f32[64]" = torch.ops.aten.clone.default(as_strided_default_344);  as_strided_default_344 = None
        as_strided_default_345: "f32[64]" = torch.ops.aten.as_strided.default(clone_default_172, [64], [1], 0);  clone_default_172 = None
        triton_kernel_wrapper_mutation_297 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 227, constant_args_idx = 324, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_55, 'SUM': as_strided_default_343, 'SUMSQ': as_strided_default_345, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_297 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_6: "f32[64]" = torch.ops.aten.div.Tensor(as_strided_default_343, full_default_8);  as_strided_default_343 = None
        div_7: "f32[64]" = torch.ops.aten.div.Tensor(as_strided_default_345, full_default_8);  as_strided_default_345 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_296 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 228, constant_args_idx = 325, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_55, 'MEAN': div_6, 'INVSTD': rsqrt_2, 'GAMMA': primals_16, 'BETA': primals_17, 'Y': permute_6, 'X_hat': permute_7, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024});  view_55 = div_6 = primals_17 = triton_kernel_wrapper_mutation_296 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_23: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_24: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_25: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_59: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(permute_7, [512, -1, 512]);  permute_7 = None
        view_60: "bf16[200704, 512]" = torch.ops.aten.reshape.default(view_59, [200704, 512]);  view_59 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_295 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 229, constant_args_idx = 326, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_60, 'P_ptr': empty_23, 'S_ptr': empty_24, 'M_ptr': empty_25, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_60 = triton_kernel_wrapper_mutation_295 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_66: "bf16[512, 64, 56, 56]" = torch.ops.aten.reshape.default(permute_6, [512, 64, 56, 56]);  permute_6 = None
        empty_26: "bf16[512, 64, 56, 56]" = torch.ops.aten.empty.memory_format([512, 64, 56, 56], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_8: "bf16[512, 64, 56, 56]" = torch.ops.aten.permute.default(empty_26, [0, 1, 2, 3]);  empty_26 = None
        
        # No stacktrace found for following nodes
        as_strided_default_340: "i8[102760448]" = torch.ops.aten.as_strided.default(full_default_10, [102760448], [1], 0)
        clone_default_170: "i8[102760448]" = torch.ops.aten.clone.default(as_strided_default_340);  as_strided_default_340 = None
        as_strided_default_341: "i8[512, 64, 56, 56]" = torch.ops.aten.as_strided.default(clone_default_170, [512, 64, 56, 56], [200704, 3136, 56, 1], 0);  clone_default_170 = None
        triton_kernel_wrapper_mutation_294 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 327, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_66, 'Y_ptr': permute_8, 'Mask_prt': as_strided_default_341, 'n_elts': 102760448, 'BLOCK_SIZE': 1024});  view_66 = triton_kernel_wrapper_mutation_294 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_69: "i8[512, 392, 512]" = torch.ops.aten.reshape.default(as_strided_default_341, [512, -1, 512]);  as_strided_default_341 = None
        view_70: "i8[200704, 512]" = torch.ops.aten.reshape.default(view_69, [200704, 512]);  view_69 = None
        
        # No stacktrace found for following nodes
        as_strided_default_338: "i32[3211264]" = torch.ops.aten.as_strided.default(full_default_11, [3211264], [1], 0)
        clone_default_169: "i32[3211264]" = torch.ops.aten.clone.default(as_strided_default_338);  as_strided_default_338 = None
        as_strided_default_339: "i32[200704, 16]" = torch.ops.aten.as_strided.default(clone_default_169, [200704, 16], [16, 1], 0);  clone_default_169 = None
        triton_kernel_wrapper_mutation_293 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 328, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_70, 'P_ptr': as_strided_default_339, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_70 = triton_kernel_wrapper_mutation_293 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_4: "bf16[256, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        empty_27: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_28: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_29: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_73: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(permute_8, [512, -1, 512])
        view_74: "bf16[200704, 512]" = torch.ops.aten.reshape.default(view_73, [200704, 512]);  view_73 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_292 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 230, constant_args_idx = 329, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_74, 'P_ptr': empty_27, 'S_ptr': empty_28, 'M_ptr': empty_29, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_74 = triton_kernel_wrapper_mutation_292 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_3: "bf16[512, 256, 56, 56]" = torch.ops.aten.convolution.default(permute_8, convert_element_type_4, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_8 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_12: "i64[]" = torch.ops.aten.add.Tensor(primals_21, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_80: "bf16[512, 256, 3136]" = torch.ops.aten.reshape.default(convolution_3, [512, 256, 3136]);  convolution_3 = None
        full_default_18: "f32[256]" = torch.ops.aten.full.default([256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        as_strided_default_334: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_167: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_334);  as_strided_default_334 = None
        as_strided_default_335: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_167, [256], [1], 0);  clone_default_167 = None
        as_strided_default_336: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_168: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_336);  as_strided_default_336 = None
        as_strided_default_337: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_168, [256], [1], 0);  clone_default_168 = None
        triton_kernel_wrapper_mutation_291 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 231, constant_args_idx = 330, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_80, 'SUM': as_strided_default_335, 'SUMSQ': as_strided_default_337, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_291 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_9: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_335, full_default_8);  as_strided_default_335 = None
        div_10: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_337, full_default_8);  as_strided_default_337 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_290 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 232, constant_args_idx = 331, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_80, 'MEAN': div_9, 'INVSTD': rsqrt_3, 'GAMMA': primals_22, 'BETA': primals_23, 'Y': permute_9, 'X_hat': permute_10, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024});  view_80 = div_9 = primals_23 = triton_kernel_wrapper_mutation_290 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_32: "i32[802816, 32]" = torch.ops.aten.empty.memory_format([802816, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_33: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_34: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_84: "bf16[512, 1568, 512]" = torch.ops.aten.reshape.default(permute_10, [512, -1, 512]);  permute_10 = None
        view_85: "bf16[802816, 512]" = torch.ops.aten.reshape.default(view_84, [802816, 512]);  view_84 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_289 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 233, constant_args_idx = 332, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_85, 'P_ptr': empty_32, 'S_ptr': empty_33, 'M_ptr': empty_34, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_85 = triton_kernel_wrapper_mutation_289 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_5: "bf16[256, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16);  primals_26 = None
        empty_35: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_36: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_37: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_288 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 234, constant_args_idx = 333, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_24, 'P_ptr': empty_35, 'S_ptr': empty_36, 'M_ptr': empty_37, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_24 = triton_kernel_wrapper_mutation_288 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_4: "bf16[512, 256, 56, 56]" = torch.ops.aten.convolution.default(getitem_13, convert_element_type_5, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_13 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_16: "i64[]" = torch.ops.aten.add.Tensor(primals_27, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_98: "bf16[512, 256, 3136]" = torch.ops.aten.reshape.default(convolution_4, [512, 256, 3136]);  convolution_4 = None
        
        # No stacktrace found for following nodes
        as_strided_default_330: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_165: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_330);  as_strided_default_330 = None
        as_strided_default_331: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_165, [256], [1], 0);  clone_default_165 = None
        as_strided_default_332: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_166: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_332);  as_strided_default_332 = None
        as_strided_default_333: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_166, [256], [1], 0);  clone_default_166 = None
        triton_kernel_wrapper_mutation_287 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 235, constant_args_idx = 334, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_98, 'SUM': as_strided_default_331, 'SUMSQ': as_strided_default_333, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_287 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_12: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_331, full_default_8);  as_strided_default_331 = None
        div_13: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_333, full_default_8);  as_strided_default_333 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_286 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 236, constant_args_idx = 335, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_98, 'MEAN': div_12, 'INVSTD': rsqrt_4, 'GAMMA': primals_28, 'BETA': primals_29, 'Y': permute_11, 'X_hat': permute_12, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024});  view_98 = div_12 = primals_29 = triton_kernel_wrapper_mutation_286 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_40: "i32[802816, 32]" = torch.ops.aten.empty.memory_format([802816, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_41: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_42: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_102: "bf16[512, 1568, 512]" = torch.ops.aten.reshape.default(permute_12, [512, -1, 512]);  permute_12 = None
        view_103: "bf16[802816, 512]" = torch.ops.aten.reshape.default(view_102, [802816, 512]);  view_102 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_285 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 237, constant_args_idx = 336, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_103, 'P_ptr': empty_40, 'S_ptr': empty_41, 'M_ptr': empty_42, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_103 = triton_kernel_wrapper_mutation_285 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:19 in forward, code: add = layer1_0_bn3 + layer1_0_downsample_1;  layer1_0_bn3 = layer1_0_downsample_1 = None
        view_109: "bf16[512, 256, 56, 56]" = torch.ops.aten.reshape.default(permute_9, [512, 256, 56, 56]);  permute_9 = None
        view_110: "bf16[512, 256, 56, 56]" = torch.ops.aten.reshape.default(permute_11, [512, 256, 56, 56]);  permute_11 = None
        add_20: "bf16[512, 256, 56, 56]" = torch.ops.aten.add.Tensor(view_109, view_110);  view_109 = view_110 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_26: "i8[512, 256, 56, 56]" = torch.ops.aten.full.default([512, 256, 56, 56], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        empty_43: "bf16[512, 256, 56, 56]" = torch.ops.aten.empty.memory_format([512, 256, 56, 56], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_13: "bf16[512, 256, 56, 56]" = torch.ops.aten.permute.default(empty_43, [0, 1, 2, 3]);  empty_43 = None
        
        # No stacktrace found for following nodes
        as_strided_default_328: "i8[411041792]" = torch.ops.aten.as_strided.default(full_default_26, [411041792], [1], 0)
        clone_default_164: "i8[411041792]" = torch.ops.aten.clone.default(as_strided_default_328);  as_strided_default_328 = None
        as_strided_default_329: "i8[512, 256, 56, 56]" = torch.ops.aten.as_strided.default(clone_default_164, [512, 256, 56, 56], [802816, 3136, 56, 1], 0);  clone_default_164 = None
        triton_kernel_wrapper_mutation_284 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 337, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_20, 'Y_ptr': permute_13, 'Mask_prt': as_strided_default_329, 'n_elts': 411041792, 'BLOCK_SIZE': 1024});  add_20 = triton_kernel_wrapper_mutation_284 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_113: "i8[512, 1568, 512]" = torch.ops.aten.reshape.default(as_strided_default_329, [512, -1, 512]);  as_strided_default_329 = None
        view_114: "i8[802816, 512]" = torch.ops.aten.reshape.default(view_113, [802816, 512]);  view_113 = None
        
        # No stacktrace found for following nodes
        as_strided_default_326: "i32[12845056]" = torch.ops.aten.as_strided.default(full_default_5, [12845056], [1], 0)
        clone_default_163: "i32[12845056]" = torch.ops.aten.clone.default(as_strided_default_326);  as_strided_default_326 = None
        as_strided_default_327: "i32[802816, 16]" = torch.ops.aten.as_strided.default(clone_default_163, [802816, 16], [16, 1], 0);  clone_default_163 = None
        triton_kernel_wrapper_mutation_283 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 338, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_114, 'P_ptr': as_strided_default_327, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_114 = triton_kernel_wrapper_mutation_283 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_6: "bf16[64, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        empty_44: "i32[802816, 32]" = torch.ops.aten.empty.memory_format([802816, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_45: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_46: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_117: "bf16[512, 1568, 512]" = torch.ops.aten.reshape.default(permute_13, [512, -1, 512])
        view_118: "bf16[802816, 512]" = torch.ops.aten.reshape.default(view_117, [802816, 512]);  view_117 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_282 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 238, constant_args_idx = 339, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_118, 'P_ptr': empty_44, 'S_ptr': empty_45, 'M_ptr': empty_46, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_118 = triton_kernel_wrapper_mutation_282 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_5: "bf16[512, 64, 56, 56]" = torch.ops.aten.convolution.default(permute_13, convert_element_type_6, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_21: "i64[]" = torch.ops.aten.add.Tensor(primals_33, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_124: "bf16[512, 64, 3136]" = torch.ops.aten.reshape.default(convolution_5, [512, 64, 3136]);  convolution_5 = None
        
        # No stacktrace found for following nodes
        as_strided_default_322: "f32[64]" = torch.ops.aten.as_strided.default(full_default, [64], [1], 0)
        clone_default_161: "f32[64]" = torch.ops.aten.clone.default(as_strided_default_322);  as_strided_default_322 = None
        as_strided_default_323: "f32[64]" = torch.ops.aten.as_strided.default(clone_default_161, [64], [1], 0);  clone_default_161 = None
        as_strided_default_324: "f32[64]" = torch.ops.aten.as_strided.default(full_default, [64], [1], 0)
        clone_default_162: "f32[64]" = torch.ops.aten.clone.default(as_strided_default_324);  as_strided_default_324 = None
        as_strided_default_325: "f32[64]" = torch.ops.aten.as_strided.default(clone_default_162, [64], [1], 0);  clone_default_162 = None
        triton_kernel_wrapper_mutation_281 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 239, constant_args_idx = 340, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_124, 'SUM': as_strided_default_323, 'SUMSQ': as_strided_default_325, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_281 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_15: "f32[64]" = torch.ops.aten.div.Tensor(as_strided_default_323, full_default_8);  as_strided_default_323 = None
        div_16: "f32[64]" = torch.ops.aten.div.Tensor(as_strided_default_325, full_default_8);  as_strided_default_325 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_280 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 240, constant_args_idx = 341, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_124, 'MEAN': div_15, 'INVSTD': rsqrt_5, 'GAMMA': primals_34, 'BETA': primals_35, 'Y': permute_14, 'X_hat': permute_15, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024});  view_124 = div_15 = primals_35 = triton_kernel_wrapper_mutation_280 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_49: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_50: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_51: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_128: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(permute_15, [512, -1, 512]);  permute_15 = None
        view_129: "bf16[200704, 512]" = torch.ops.aten.reshape.default(view_128, [200704, 512]);  view_128 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_279 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 241, constant_args_idx = 342, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_129, 'P_ptr': empty_49, 'S_ptr': empty_50, 'M_ptr': empty_51, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_129 = triton_kernel_wrapper_mutation_279 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_135: "bf16[512, 64, 56, 56]" = torch.ops.aten.reshape.default(permute_14, [512, 64, 56, 56]);  permute_14 = None
        empty_52: "bf16[512, 64, 56, 56]" = torch.ops.aten.empty.memory_format([512, 64, 56, 56], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_16: "bf16[512, 64, 56, 56]" = torch.ops.aten.permute.default(empty_52, [0, 1, 2, 3]);  empty_52 = None
        
        # No stacktrace found for following nodes
        as_strided_default_320: "i8[102760448]" = torch.ops.aten.as_strided.default(full_default_10, [102760448], [1], 0)
        clone_default_160: "i8[102760448]" = torch.ops.aten.clone.default(as_strided_default_320);  as_strided_default_320 = None
        as_strided_default_321: "i8[512, 64, 56, 56]" = torch.ops.aten.as_strided.default(clone_default_160, [512, 64, 56, 56], [200704, 3136, 56, 1], 0);  clone_default_160 = None
        triton_kernel_wrapper_mutation_278 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 343, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_135, 'Y_ptr': permute_16, 'Mask_prt': as_strided_default_321, 'n_elts': 102760448, 'BLOCK_SIZE': 1024});  view_135 = triton_kernel_wrapper_mutation_278 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_138: "i8[512, 392, 512]" = torch.ops.aten.reshape.default(as_strided_default_321, [512, -1, 512]);  as_strided_default_321 = None
        view_139: "i8[200704, 512]" = torch.ops.aten.reshape.default(view_138, [200704, 512]);  view_138 = None
        
        # No stacktrace found for following nodes
        as_strided_default_318: "i32[3211264]" = torch.ops.aten.as_strided.default(full_default_11, [3211264], [1], 0)
        clone_default_159: "i32[3211264]" = torch.ops.aten.clone.default(as_strided_default_318);  as_strided_default_318 = None
        as_strided_default_319: "i32[200704, 16]" = torch.ops.aten.as_strided.default(clone_default_159, [200704, 16], [16, 1], 0);  clone_default_159 = None
        triton_kernel_wrapper_mutation_277 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 344, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_139, 'P_ptr': as_strided_default_319, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_139 = triton_kernel_wrapper_mutation_277 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_7: "bf16[64, 64, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        empty_53: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_54: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_55: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_142: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(permute_16, [512, -1, 512])
        view_143: "bf16[200704, 512]" = torch.ops.aten.reshape.default(view_142, [200704, 512]);  view_142 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_276 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 242, constant_args_idx = 345, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_143, 'P_ptr': empty_53, 'S_ptr': empty_54, 'M_ptr': empty_55, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_143 = triton_kernel_wrapper_mutation_276 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_6: "bf16[512, 64, 56, 56]" = torch.ops.aten.convolution.default(permute_16, convert_element_type_7, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  permute_16 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_25: "i64[]" = torch.ops.aten.add.Tensor(primals_39, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_149: "bf16[512, 64, 3136]" = torch.ops.aten.reshape.default(convolution_6, [512, 64, 3136]);  convolution_6 = None
        
        # No stacktrace found for following nodes
        as_strided_default_314: "f32[64]" = torch.ops.aten.as_strided.default(full_default, [64], [1], 0)
        clone_default_157: "f32[64]" = torch.ops.aten.clone.default(as_strided_default_314);  as_strided_default_314 = None
        as_strided_default_315: "f32[64]" = torch.ops.aten.as_strided.default(clone_default_157, [64], [1], 0);  clone_default_157 = None
        as_strided_default_316: "f32[64]" = torch.ops.aten.as_strided.default(full_default, [64], [1], 0)
        clone_default_158: "f32[64]" = torch.ops.aten.clone.default(as_strided_default_316);  as_strided_default_316 = None
        as_strided_default_317: "f32[64]" = torch.ops.aten.as_strided.default(clone_default_158, [64], [1], 0);  clone_default_158 = None
        triton_kernel_wrapper_mutation_275 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 243, constant_args_idx = 346, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_149, 'SUM': as_strided_default_315, 'SUMSQ': as_strided_default_317, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_275 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_18: "f32[64]" = torch.ops.aten.div.Tensor(as_strided_default_315, full_default_8);  as_strided_default_315 = None
        div_19: "f32[64]" = torch.ops.aten.div.Tensor(as_strided_default_317, full_default_8);  as_strided_default_317 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_274 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 244, constant_args_idx = 347, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_149, 'MEAN': div_18, 'INVSTD': rsqrt_6, 'GAMMA': primals_40, 'BETA': primals_41, 'Y': permute_17, 'X_hat': permute_18, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024});  view_149 = div_18 = primals_41 = triton_kernel_wrapper_mutation_274 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_58: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_59: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_60: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_153: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(permute_18, [512, -1, 512]);  permute_18 = None
        view_154: "bf16[200704, 512]" = torch.ops.aten.reshape.default(view_153, [200704, 512]);  view_153 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_273 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 245, constant_args_idx = 348, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_154, 'P_ptr': empty_58, 'S_ptr': empty_59, 'M_ptr': empty_60, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_154 = triton_kernel_wrapper_mutation_273 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_160: "bf16[512, 64, 56, 56]" = torch.ops.aten.reshape.default(permute_17, [512, 64, 56, 56]);  permute_17 = None
        empty_61: "bf16[512, 64, 56, 56]" = torch.ops.aten.empty.memory_format([512, 64, 56, 56], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_19: "bf16[512, 64, 56, 56]" = torch.ops.aten.permute.default(empty_61, [0, 1, 2, 3]);  empty_61 = None
        
        # No stacktrace found for following nodes
        as_strided_default_312: "i8[102760448]" = torch.ops.aten.as_strided.default(full_default_10, [102760448], [1], 0)
        clone_default_156: "i8[102760448]" = torch.ops.aten.clone.default(as_strided_default_312);  as_strided_default_312 = None
        as_strided_default_313: "i8[512, 64, 56, 56]" = torch.ops.aten.as_strided.default(clone_default_156, [512, 64, 56, 56], [200704, 3136, 56, 1], 0);  clone_default_156 = None
        triton_kernel_wrapper_mutation_272 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 349, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_160, 'Y_ptr': permute_19, 'Mask_prt': as_strided_default_313, 'n_elts': 102760448, 'BLOCK_SIZE': 1024});  view_160 = triton_kernel_wrapper_mutation_272 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_163: "i8[512, 392, 512]" = torch.ops.aten.reshape.default(as_strided_default_313, [512, -1, 512]);  as_strided_default_313 = None
        view_164: "i8[200704, 512]" = torch.ops.aten.reshape.default(view_163, [200704, 512]);  view_163 = None
        
        # No stacktrace found for following nodes
        as_strided_default_310: "i32[3211264]" = torch.ops.aten.as_strided.default(full_default_11, [3211264], [1], 0)
        clone_default_155: "i32[3211264]" = torch.ops.aten.clone.default(as_strided_default_310);  as_strided_default_310 = None
        as_strided_default_311: "i32[200704, 16]" = torch.ops.aten.as_strided.default(clone_default_155, [200704, 16], [16, 1], 0);  clone_default_155 = None
        triton_kernel_wrapper_mutation_271 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 350, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_164, 'P_ptr': as_strided_default_311, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_164 = triton_kernel_wrapper_mutation_271 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_8: "bf16[256, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        empty_62: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_63: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_64: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_167: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(permute_19, [512, -1, 512])
        view_168: "bf16[200704, 512]" = torch.ops.aten.reshape.default(view_167, [200704, 512]);  view_167 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_270 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 246, constant_args_idx = 351, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_168, 'P_ptr': empty_62, 'S_ptr': empty_63, 'M_ptr': empty_64, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_168 = triton_kernel_wrapper_mutation_270 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_7: "bf16[512, 256, 56, 56]" = torch.ops.aten.convolution.default(permute_19, convert_element_type_8, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_19 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_29: "i64[]" = torch.ops.aten.add.Tensor(primals_45, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_174: "bf16[512, 256, 3136]" = torch.ops.aten.reshape.default(convolution_7, [512, 256, 3136]);  convolution_7 = None
        
        # No stacktrace found for following nodes
        as_strided_default_306: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_153: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_306);  as_strided_default_306 = None
        as_strided_default_307: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_153, [256], [1], 0);  clone_default_153 = None
        as_strided_default_308: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_154: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_308);  as_strided_default_308 = None
        as_strided_default_309: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_154, [256], [1], 0);  clone_default_154 = None
        triton_kernel_wrapper_mutation_269 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 247, constant_args_idx = 352, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_174, 'SUM': as_strided_default_307, 'SUMSQ': as_strided_default_309, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_269 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_21: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_307, full_default_8);  as_strided_default_307 = None
        div_22: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_309, full_default_8);  as_strided_default_309 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_268 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 248, constant_args_idx = 353, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_174, 'MEAN': div_21, 'INVSTD': rsqrt_7, 'GAMMA': primals_46, 'BETA': primals_47, 'Y': permute_20, 'X_hat': permute_21, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024});  view_174 = div_21 = primals_47 = triton_kernel_wrapper_mutation_268 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_67: "i32[802816, 32]" = torch.ops.aten.empty.memory_format([802816, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_68: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_69: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_178: "bf16[512, 1568, 512]" = torch.ops.aten.reshape.default(permute_21, [512, -1, 512]);  permute_21 = None
        view_179: "bf16[802816, 512]" = torch.ops.aten.reshape.default(view_178, [802816, 512]);  view_178 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_267 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 249, constant_args_idx = 354, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_179, 'P_ptr': empty_67, 'S_ptr': empty_68, 'M_ptr': empty_69, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_179 = triton_kernel_wrapper_mutation_267 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:29 in forward, code: add_1 = layer1_1_bn3 + layer1_0_relu_2;  layer1_1_bn3 = layer1_0_relu_2 = None
        view_185: "bf16[512, 256, 56, 56]" = torch.ops.aten.reshape.default(permute_20, [512, 256, 56, 56]);  permute_20 = None
        add_33: "bf16[512, 256, 56, 56]" = torch.ops.aten.add.Tensor(view_185, permute_13);  view_185 = permute_13 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        empty_70: "bf16[512, 256, 56, 56]" = torch.ops.aten.empty.memory_format([512, 256, 56, 56], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_22: "bf16[512, 256, 56, 56]" = torch.ops.aten.permute.default(empty_70, [0, 1, 2, 3]);  empty_70 = None
        
        # No stacktrace found for following nodes
        as_strided_default_304: "i8[411041792]" = torch.ops.aten.as_strided.default(full_default_26, [411041792], [1], 0)
        clone_default_152: "i8[411041792]" = torch.ops.aten.clone.default(as_strided_default_304);  as_strided_default_304 = None
        as_strided_default_305: "i8[512, 256, 56, 56]" = torch.ops.aten.as_strided.default(clone_default_152, [512, 256, 56, 56], [802816, 3136, 56, 1], 0);  clone_default_152 = None
        triton_kernel_wrapper_mutation_266 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 355, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_33, 'Y_ptr': permute_22, 'Mask_prt': as_strided_default_305, 'n_elts': 411041792, 'BLOCK_SIZE': 1024});  add_33 = triton_kernel_wrapper_mutation_266 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_188: "i8[512, 1568, 512]" = torch.ops.aten.reshape.default(as_strided_default_305, [512, -1, 512]);  as_strided_default_305 = None
        view_189: "i8[802816, 512]" = torch.ops.aten.reshape.default(view_188, [802816, 512]);  view_188 = None
        
        # No stacktrace found for following nodes
        as_strided_default_302: "i32[12845056]" = torch.ops.aten.as_strided.default(full_default_5, [12845056], [1], 0)
        clone_default_151: "i32[12845056]" = torch.ops.aten.clone.default(as_strided_default_302);  as_strided_default_302 = None
        as_strided_default_303: "i32[802816, 16]" = torch.ops.aten.as_strided.default(clone_default_151, [802816, 16], [16, 1], 0);  clone_default_151 = None
        triton_kernel_wrapper_mutation_265 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 356, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_189, 'P_ptr': as_strided_default_303, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_189 = triton_kernel_wrapper_mutation_265 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_9: "bf16[64, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16);  primals_50 = None
        empty_71: "i32[802816, 32]" = torch.ops.aten.empty.memory_format([802816, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_72: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_73: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_192: "bf16[512, 1568, 512]" = torch.ops.aten.reshape.default(permute_22, [512, -1, 512])
        view_193: "bf16[802816, 512]" = torch.ops.aten.reshape.default(view_192, [802816, 512]);  view_192 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_264 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 250, constant_args_idx = 357, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_193, 'P_ptr': empty_71, 'S_ptr': empty_72, 'M_ptr': empty_73, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_193 = triton_kernel_wrapper_mutation_264 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_8: "bf16[512, 64, 56, 56]" = torch.ops.aten.convolution.default(permute_22, convert_element_type_9, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_34: "i64[]" = torch.ops.aten.add.Tensor(primals_51, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_199: "bf16[512, 64, 3136]" = torch.ops.aten.reshape.default(convolution_8, [512, 64, 3136]);  convolution_8 = None
        
        # No stacktrace found for following nodes
        as_strided_default_298: "f32[64]" = torch.ops.aten.as_strided.default(full_default, [64], [1], 0)
        clone_default_149: "f32[64]" = torch.ops.aten.clone.default(as_strided_default_298);  as_strided_default_298 = None
        as_strided_default_299: "f32[64]" = torch.ops.aten.as_strided.default(clone_default_149, [64], [1], 0);  clone_default_149 = None
        as_strided_default_300: "f32[64]" = torch.ops.aten.as_strided.default(full_default, [64], [1], 0)
        clone_default_150: "f32[64]" = torch.ops.aten.clone.default(as_strided_default_300);  as_strided_default_300 = None
        as_strided_default_301: "f32[64]" = torch.ops.aten.as_strided.default(clone_default_150, [64], [1], 0);  clone_default_150 = None
        triton_kernel_wrapper_mutation_263 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 251, constant_args_idx = 358, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_199, 'SUM': as_strided_default_299, 'SUMSQ': as_strided_default_301, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_263 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_24: "f32[64]" = torch.ops.aten.div.Tensor(as_strided_default_299, full_default_8);  as_strided_default_299 = None
        div_25: "f32[64]" = torch.ops.aten.div.Tensor(as_strided_default_301, full_default_8);  as_strided_default_301 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_262 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 252, constant_args_idx = 359, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_199, 'MEAN': div_24, 'INVSTD': rsqrt_8, 'GAMMA': primals_52, 'BETA': primals_53, 'Y': permute_23, 'X_hat': permute_24, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024});  view_199 = div_24 = primals_53 = triton_kernel_wrapper_mutation_262 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_76: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_77: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_78: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_203: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(permute_24, [512, -1, 512]);  permute_24 = None
        view_204: "bf16[200704, 512]" = torch.ops.aten.reshape.default(view_203, [200704, 512]);  view_203 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_261 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 253, constant_args_idx = 360, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_204, 'P_ptr': empty_76, 'S_ptr': empty_77, 'M_ptr': empty_78, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_204 = triton_kernel_wrapper_mutation_261 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_210: "bf16[512, 64, 56, 56]" = torch.ops.aten.reshape.default(permute_23, [512, 64, 56, 56]);  permute_23 = None
        empty_79: "bf16[512, 64, 56, 56]" = torch.ops.aten.empty.memory_format([512, 64, 56, 56], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_25: "bf16[512, 64, 56, 56]" = torch.ops.aten.permute.default(empty_79, [0, 1, 2, 3]);  empty_79 = None
        
        # No stacktrace found for following nodes
        as_strided_default_296: "i8[102760448]" = torch.ops.aten.as_strided.default(full_default_10, [102760448], [1], 0)
        clone_default_148: "i8[102760448]" = torch.ops.aten.clone.default(as_strided_default_296);  as_strided_default_296 = None
        as_strided_default_297: "i8[512, 64, 56, 56]" = torch.ops.aten.as_strided.default(clone_default_148, [512, 64, 56, 56], [200704, 3136, 56, 1], 0);  clone_default_148 = None
        triton_kernel_wrapper_mutation_260 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 361, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_210, 'Y_ptr': permute_25, 'Mask_prt': as_strided_default_297, 'n_elts': 102760448, 'BLOCK_SIZE': 1024});  view_210 = triton_kernel_wrapper_mutation_260 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_213: "i8[512, 392, 512]" = torch.ops.aten.reshape.default(as_strided_default_297, [512, -1, 512]);  as_strided_default_297 = None
        view_214: "i8[200704, 512]" = torch.ops.aten.reshape.default(view_213, [200704, 512]);  view_213 = None
        
        # No stacktrace found for following nodes
        as_strided_default_294: "i32[3211264]" = torch.ops.aten.as_strided.default(full_default_11, [3211264], [1], 0)
        clone_default_147: "i32[3211264]" = torch.ops.aten.clone.default(as_strided_default_294);  as_strided_default_294 = None
        as_strided_default_295: "i32[200704, 16]" = torch.ops.aten.as_strided.default(clone_default_147, [200704, 16], [16, 1], 0);  clone_default_147 = None
        triton_kernel_wrapper_mutation_259 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 362, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_214, 'P_ptr': as_strided_default_295, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_214 = triton_kernel_wrapper_mutation_259 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_10: "bf16[64, 64, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_56, torch.bfloat16);  primals_56 = None
        empty_80: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_81: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_82: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_217: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(permute_25, [512, -1, 512])
        view_218: "bf16[200704, 512]" = torch.ops.aten.reshape.default(view_217, [200704, 512]);  view_217 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_258 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 254, constant_args_idx = 363, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_218, 'P_ptr': empty_80, 'S_ptr': empty_81, 'M_ptr': empty_82, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_218 = triton_kernel_wrapper_mutation_258 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_9: "bf16[512, 64, 56, 56]" = torch.ops.aten.convolution.default(permute_25, convert_element_type_10, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  permute_25 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_38: "i64[]" = torch.ops.aten.add.Tensor(primals_57, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_224: "bf16[512, 64, 3136]" = torch.ops.aten.reshape.default(convolution_9, [512, 64, 3136]);  convolution_9 = None
        
        # No stacktrace found for following nodes
        as_strided_default_292: "f32[64]" = torch.ops.aten.as_strided.default(full_default, [64], [1], 0)
        clone_default_146: "f32[64]" = torch.ops.aten.clone.default(as_strided_default_292);  as_strided_default_292 = None
        as_strided_default_293: "f32[64]" = torch.ops.aten.as_strided.default(clone_default_146, [64], [1], 0);  clone_default_146 = None
        triton_kernel_wrapper_mutation_257 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 255, constant_args_idx = 364, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_224, 'SUM': full_default, 'SUMSQ': as_strided_default_293, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_257 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_27: "f32[64]" = torch.ops.aten.div.Tensor(full_default, full_default_8);  full_default = None
        div_28: "f32[64]" = torch.ops.aten.div.Tensor(as_strided_default_293, full_default_8);  as_strided_default_293 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_256 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 256, constant_args_idx = 365, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_224, 'MEAN': div_27, 'INVSTD': rsqrt_9, 'GAMMA': primals_58, 'BETA': primals_59, 'Y': permute_26, 'X_hat': permute_27, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024});  view_224 = div_27 = primals_59 = triton_kernel_wrapper_mutation_256 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_85: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_86: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_87: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_228: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(permute_27, [512, -1, 512]);  permute_27 = None
        view_229: "bf16[200704, 512]" = torch.ops.aten.reshape.default(view_228, [200704, 512]);  view_228 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_255 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 257, constant_args_idx = 366, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_229, 'P_ptr': empty_85, 'S_ptr': empty_86, 'M_ptr': empty_87, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_229 = triton_kernel_wrapper_mutation_255 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_235: "bf16[512, 64, 56, 56]" = torch.ops.aten.reshape.default(permute_26, [512, 64, 56, 56]);  permute_26 = None
        empty_88: "bf16[512, 64, 56, 56]" = torch.ops.aten.empty.memory_format([512, 64, 56, 56], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_28: "bf16[512, 64, 56, 56]" = torch.ops.aten.permute.default(empty_88, [0, 1, 2, 3]);  empty_88 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_254 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 367, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_235, 'Y_ptr': permute_28, 'Mask_prt': full_default_10, 'n_elts': 102760448, 'BLOCK_SIZE': 1024});  view_235 = triton_kernel_wrapper_mutation_254 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_238: "i8[512, 392, 512]" = torch.ops.aten.reshape.default(full_default_10, [512, -1, 512]);  full_default_10 = None
        view_239: "i8[200704, 512]" = torch.ops.aten.reshape.default(view_238, [200704, 512]);  view_238 = None
        
        # No stacktrace found for following nodes
        as_strided_default_290: "i32[3211264]" = torch.ops.aten.as_strided.default(full_default_11, [3211264], [1], 0)
        clone_default_145: "i32[3211264]" = torch.ops.aten.clone.default(as_strided_default_290);  as_strided_default_290 = None
        as_strided_default_291: "i32[200704, 16]" = torch.ops.aten.as_strided.default(clone_default_145, [200704, 16], [16, 1], 0);  clone_default_145 = None
        triton_kernel_wrapper_mutation_253 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 368, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_239, 'P_ptr': as_strided_default_291, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_239 = triton_kernel_wrapper_mutation_253 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_11: "bf16[256, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16);  primals_62 = None
        empty_89: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_90: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_91: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_242: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(permute_28, [512, -1, 512])
        view_243: "bf16[200704, 512]" = torch.ops.aten.reshape.default(view_242, [200704, 512]);  view_242 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_252 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 258, constant_args_idx = 369, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_243, 'P_ptr': empty_89, 'S_ptr': empty_90, 'M_ptr': empty_91, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_243 = triton_kernel_wrapper_mutation_252 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_10: "bf16[512, 256, 56, 56]" = torch.ops.aten.convolution.default(permute_28, convert_element_type_11, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_28 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_42: "i64[]" = torch.ops.aten.add.Tensor(primals_63, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_249: "bf16[512, 256, 3136]" = torch.ops.aten.reshape.default(convolution_10, [512, 256, 3136]);  convolution_10 = None
        
        # No stacktrace found for following nodes
        as_strided_default_286: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_143: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_286);  as_strided_default_286 = None
        as_strided_default_287: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_143, [256], [1], 0);  clone_default_143 = None
        as_strided_default_288: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_144: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_288);  as_strided_default_288 = None
        as_strided_default_289: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_144, [256], [1], 0);  clone_default_144 = None
        triton_kernel_wrapper_mutation_251 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 259, constant_args_idx = 370, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_249, 'SUM': as_strided_default_287, 'SUMSQ': as_strided_default_289, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_251 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_30: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_287, full_default_8);  as_strided_default_287 = None
        div_31: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_289, full_default_8);  as_strided_default_289 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_250 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 260, constant_args_idx = 371, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_249, 'MEAN': div_30, 'INVSTD': rsqrt_10, 'GAMMA': primals_64, 'BETA': primals_65, 'Y': permute_29, 'X_hat': permute_30, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024});  view_249 = div_30 = primals_65 = triton_kernel_wrapper_mutation_250 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_94: "i32[802816, 32]" = torch.ops.aten.empty.memory_format([802816, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_95: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_96: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_253: "bf16[512, 1568, 512]" = torch.ops.aten.reshape.default(permute_30, [512, -1, 512]);  permute_30 = None
        view_254: "bf16[802816, 512]" = torch.ops.aten.reshape.default(view_253, [802816, 512]);  view_253 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_249 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 261, constant_args_idx = 372, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_254, 'P_ptr': empty_94, 'S_ptr': empty_95, 'M_ptr': empty_96, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_254 = triton_kernel_wrapper_mutation_249 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:39 in forward, code: add_2 = layer1_2_bn3 + layer1_1_relu_2;  layer1_2_bn3 = layer1_1_relu_2 = None
        view_260: "bf16[512, 256, 56, 56]" = torch.ops.aten.reshape.default(permute_29, [512, 256, 56, 56]);  permute_29 = None
        add_46: "bf16[512, 256, 56, 56]" = torch.ops.aten.add.Tensor(view_260, permute_22);  view_260 = permute_22 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        empty_97: "bf16[512, 256, 56, 56]" = torch.ops.aten.empty.memory_format([512, 256, 56, 56], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_31: "bf16[512, 256, 56, 56]" = torch.ops.aten.permute.default(empty_97, [0, 1, 2, 3]);  empty_97 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_248 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 373, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_46, 'Y_ptr': permute_31, 'Mask_prt': full_default_26, 'n_elts': 411041792, 'BLOCK_SIZE': 1024});  add_46 = triton_kernel_wrapper_mutation_248 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_263: "i8[512, 1568, 512]" = torch.ops.aten.reshape.default(full_default_26, [512, -1, 512]);  full_default_26 = None
        view_264: "i8[802816, 512]" = torch.ops.aten.reshape.default(view_263, [802816, 512]);  view_263 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_247 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 374, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_264, 'P_ptr': full_default_5, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_264 = triton_kernel_wrapper_mutation_247 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_12: "bf16[128, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_68, torch.bfloat16);  primals_68 = None
        empty_98: "i32[802816, 32]" = torch.ops.aten.empty.memory_format([802816, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_99: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_100: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_267: "bf16[512, 1568, 512]" = torch.ops.aten.reshape.default(permute_31, [512, -1, 512])
        view_268: "bf16[802816, 512]" = torch.ops.aten.reshape.default(view_267, [802816, 512]);  view_267 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_246 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 262, constant_args_idx = 375, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_268, 'P_ptr': empty_98, 'S_ptr': empty_99, 'M_ptr': empty_100, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  triton_kernel_wrapper_mutation_246 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_11: "bf16[512, 128, 56, 56]" = torch.ops.aten.convolution.default(permute_31, convert_element_type_12, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_47: "i64[]" = torch.ops.aten.add.Tensor(primals_69, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_274: "bf16[512, 128, 3136]" = torch.ops.aten.reshape.default(convolution_11, [512, 128, 3136]);  convolution_11 = None
        full_default_64: "f32[128]" = torch.ops.aten.full.default([128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        as_strided_default_282: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_141: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_282);  as_strided_default_282 = None
        as_strided_default_283: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_141, [128], [1], 0);  clone_default_141 = None
        as_strided_default_284: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_142: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_284);  as_strided_default_284 = None
        as_strided_default_285: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_142, [128], [1], 0);  clone_default_142 = None
        triton_kernel_wrapper_mutation_245 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 263, constant_args_idx = 376, grid = [(128, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_274, 'SUM': as_strided_default_283, 'SUMSQ': as_strided_default_285, 'M': 1605632, 'HW': 3136, 'stride_n': 401408, 'stride_c': 3136, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_245 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_33: "f32[128]" = torch.ops.aten.div.Tensor(as_strided_default_283, full_default_8);  as_strided_default_283 = None
        div_34: "f32[128]" = torch.ops.aten.div.Tensor(as_strided_default_285, full_default_8);  as_strided_default_285 = full_default_8 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_244 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 264, constant_args_idx = 377, grid = [(128, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_274, 'MEAN': div_33, 'INVSTD': rsqrt_11, 'GAMMA': primals_70, 'BETA': primals_71, 'Y': permute_32, 'X_hat': permute_33, 'M': 1605632, 'HW': 3136, 'stride_n': 401408, 'stride_c': 3136, 'BLOCK_M': 1024});  view_274 = div_33 = primals_71 = triton_kernel_wrapper_mutation_244 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_103: "i32[401408, 32]" = torch.ops.aten.empty.memory_format([401408, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_104: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_105: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_278: "bf16[512, 784, 512]" = torch.ops.aten.reshape.default(permute_33, [512, -1, 512]);  permute_33 = None
        view_279: "bf16[401408, 512]" = torch.ops.aten.reshape.default(view_278, [401408, 512]);  view_278 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_243 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 265, constant_args_idx = 378, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_279, 'P_ptr': empty_103, 'S_ptr': empty_104, 'M_ptr': empty_105, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_279 = triton_kernel_wrapper_mutation_243 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_68: "i8[512, 128, 56, 56]" = torch.ops.aten.full.default([512, 128, 56, 56], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_285: "bf16[512, 128, 56, 56]" = torch.ops.aten.reshape.default(permute_32, [512, 128, 56, 56]);  permute_32 = None
        empty_106: "bf16[512, 128, 56, 56]" = torch.ops.aten.empty.memory_format([512, 128, 56, 56], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_34: "bf16[512, 128, 56, 56]" = torch.ops.aten.permute.default(empty_106, [0, 1, 2, 3]);  empty_106 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_242 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 379, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_285, 'Y_ptr': permute_34, 'Mask_prt': full_default_68, 'n_elts': 205520896, 'BLOCK_SIZE': 1024});  view_285 = triton_kernel_wrapper_mutation_242 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_69: "i32[401408, 16]" = torch.ops.aten.full.default([401408, 16], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_288: "i8[512, 784, 512]" = torch.ops.aten.reshape.default(full_default_68, [512, -1, 512]);  full_default_68 = None
        view_289: "i8[401408, 512]" = torch.ops.aten.reshape.default(view_288, [401408, 512]);  view_288 = None
        
        # No stacktrace found for following nodes
        as_strided_default_280: "i32[6422528]" = torch.ops.aten.as_strided.default(full_default_69, [6422528], [1], 0)
        clone_default_140: "i32[6422528]" = torch.ops.aten.clone.default(as_strided_default_280);  as_strided_default_280 = None
        as_strided_default_281: "i32[401408, 16]" = torch.ops.aten.as_strided.default(clone_default_140, [401408, 16], [16, 1], 0);  clone_default_140 = None
        triton_kernel_wrapper_mutation_241 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 380, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_289, 'P_ptr': as_strided_default_281, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_289 = triton_kernel_wrapper_mutation_241 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_13: "bf16[128, 128, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_74, torch.bfloat16);  primals_74 = None
        empty_107: "i32[401408, 32]" = torch.ops.aten.empty.memory_format([401408, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_108: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_109: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_292: "bf16[512, 784, 512]" = torch.ops.aten.reshape.default(permute_34, [512, -1, 512])
        view_293: "bf16[401408, 512]" = torch.ops.aten.reshape.default(view_292, [401408, 512]);  view_292 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_240 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 266, constant_args_idx = 381, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_293, 'P_ptr': empty_107, 'S_ptr': empty_108, 'M_ptr': empty_109, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_293 = triton_kernel_wrapper_mutation_240 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_12: "bf16[512, 128, 28, 28]" = torch.ops.aten.convolution.default(permute_34, convert_element_type_13, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  permute_34 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_51: "i64[]" = torch.ops.aten.add.Tensor(primals_75, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_299: "bf16[512, 128, 784]" = torch.ops.aten.reshape.default(convolution_12, [512, 128, 784]);  convolution_12 = None
        
        # No stacktrace found for following nodes
        as_strided_default_276: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_138: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_276);  as_strided_default_276 = None
        as_strided_default_277: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_138, [128], [1], 0);  clone_default_138 = None
        as_strided_default_278: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_139: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_278);  as_strided_default_278 = None
        as_strided_default_279: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_139, [128], [1], 0);  clone_default_139 = None
        triton_kernel_wrapper_mutation_239 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 267, constant_args_idx = 382, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_299, 'SUM': as_strided_default_277, 'SUMSQ': as_strided_default_279, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_239 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        full_default_72: "f32[]" = torch.ops.aten.full.default([], 401408.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        div_36: "f32[128]" = torch.ops.aten.div.Tensor(as_strided_default_277, full_default_72);  as_strided_default_277 = None
        div_37: "f32[128]" = torch.ops.aten.div.Tensor(as_strided_default_279, full_default_72);  as_strided_default_279 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_238 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 268, constant_args_idx = 383, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_299, 'MEAN': div_36, 'INVSTD': rsqrt_12, 'GAMMA': primals_76, 'BETA': primals_77, 'Y': permute_35, 'X_hat': permute_36, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024});  view_299 = div_36 = primals_77 = triton_kernel_wrapper_mutation_238 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_112: "i32[100352, 32]" = torch.ops.aten.empty.memory_format([100352, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_113: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_114: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_303: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(permute_36, [512, -1, 512]);  permute_36 = None
        view_304: "bf16[100352, 512]" = torch.ops.aten.reshape.default(view_303, [100352, 512]);  view_303 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_237 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 269, constant_args_idx = 384, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_304, 'P_ptr': empty_112, 'S_ptr': empty_113, 'M_ptr': empty_114, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_304 = triton_kernel_wrapper_mutation_237 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_74: "i8[512, 128, 28, 28]" = torch.ops.aten.full.default([512, 128, 28, 28], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_310: "bf16[512, 128, 28, 28]" = torch.ops.aten.reshape.default(permute_35, [512, 128, 28, 28]);  permute_35 = None
        empty_115: "bf16[512, 128, 28, 28]" = torch.ops.aten.empty.memory_format([512, 128, 28, 28], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_37: "bf16[512, 128, 28, 28]" = torch.ops.aten.permute.default(empty_115, [0, 1, 2, 3]);  empty_115 = None
        
        # No stacktrace found for following nodes
        as_strided_default_274: "i8[51380224]" = torch.ops.aten.as_strided.default(full_default_74, [51380224], [1], 0)
        clone_default_137: "i8[51380224]" = torch.ops.aten.clone.default(as_strided_default_274);  as_strided_default_274 = None
        as_strided_default_275: "i8[512, 128, 28, 28]" = torch.ops.aten.as_strided.default(clone_default_137, [512, 128, 28, 28], [100352, 784, 28, 1], 0);  clone_default_137 = None
        triton_kernel_wrapper_mutation_236 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 385, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_310, 'Y_ptr': permute_37, 'Mask_prt': as_strided_default_275, 'n_elts': 51380224, 'BLOCK_SIZE': 1024});  view_310 = triton_kernel_wrapper_mutation_236 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_75: "i32[100352, 16]" = torch.ops.aten.full.default([100352, 16], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_313: "i8[512, 196, 512]" = torch.ops.aten.reshape.default(as_strided_default_275, [512, -1, 512]);  as_strided_default_275 = None
        view_314: "i8[100352, 512]" = torch.ops.aten.reshape.default(view_313, [100352, 512]);  view_313 = None
        
        # No stacktrace found for following nodes
        as_strided_default_272: "i32[1605632]" = torch.ops.aten.as_strided.default(full_default_75, [1605632], [1], 0)
        clone_default_136: "i32[1605632]" = torch.ops.aten.clone.default(as_strided_default_272);  as_strided_default_272 = None
        as_strided_default_273: "i32[100352, 16]" = torch.ops.aten.as_strided.default(clone_default_136, [100352, 16], [16, 1], 0);  clone_default_136 = None
        triton_kernel_wrapper_mutation_235 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 386, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_314, 'P_ptr': as_strided_default_273, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_314 = triton_kernel_wrapper_mutation_235 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_14: "bf16[512, 128, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_80, torch.bfloat16);  primals_80 = None
        empty_116: "i32[100352, 32]" = torch.ops.aten.empty.memory_format([100352, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_117: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_118: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_317: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(permute_37, [512, -1, 512])
        view_318: "bf16[100352, 512]" = torch.ops.aten.reshape.default(view_317, [100352, 512]);  view_317 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_234 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 270, constant_args_idx = 387, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_318, 'P_ptr': empty_116, 'S_ptr': empty_117, 'M_ptr': empty_118, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_318 = triton_kernel_wrapper_mutation_234 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_13: "bf16[512, 512, 28, 28]" = torch.ops.aten.convolution.default(permute_37, convert_element_type_14, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_37 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_55: "i64[]" = torch.ops.aten.add.Tensor(primals_81, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_324: "bf16[512, 512, 784]" = torch.ops.aten.reshape.default(convolution_13, [512, 512, 784]);  convolution_13 = None
        full_default_76: "f32[512]" = torch.ops.aten.full.default([512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        as_strided_default_268: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_134: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_268);  as_strided_default_268 = None
        as_strided_default_269: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_134, [512], [1], 0);  clone_default_134 = None
        as_strided_default_270: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_135: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_270);  as_strided_default_270 = None
        as_strided_default_271: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_135, [512], [1], 0);  clone_default_135 = None
        triton_kernel_wrapper_mutation_233 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 271, constant_args_idx = 388, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_324, 'SUM': as_strided_default_269, 'SUMSQ': as_strided_default_271, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_233 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_39: "f32[512]" = torch.ops.aten.div.Tensor(as_strided_default_269, full_default_72);  as_strided_default_269 = None
        div_40: "f32[512]" = torch.ops.aten.div.Tensor(as_strided_default_271, full_default_72);  as_strided_default_271 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_232 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 272, constant_args_idx = 389, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_324, 'MEAN': div_39, 'INVSTD': rsqrt_13, 'GAMMA': primals_82, 'BETA': primals_83, 'Y': permute_38, 'X_hat': permute_39, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024});  view_324 = div_39 = primals_83 = triton_kernel_wrapper_mutation_232 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_121: "i32[401408, 32]" = torch.ops.aten.empty.memory_format([401408, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_122: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_123: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_328: "bf16[512, 784, 512]" = torch.ops.aten.reshape.default(permute_39, [512, -1, 512]);  permute_39 = None
        view_329: "bf16[401408, 512]" = torch.ops.aten.reshape.default(view_328, [401408, 512]);  view_328 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_231 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 273, constant_args_idx = 390, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_329, 'P_ptr': empty_121, 'S_ptr': empty_122, 'M_ptr': empty_123, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_329 = triton_kernel_wrapper_mutation_231 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_15: "bf16[512, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_86, torch.bfloat16);  primals_86 = None
        empty_124: "i32[802816, 32]" = torch.ops.aten.empty.memory_format([802816, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_125: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_126: "bf16[802816]" = torch.ops.aten.empty.memory_format([802816], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_230 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 274, constant_args_idx = 391, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_268, 'P_ptr': empty_124, 'S_ptr': empty_125, 'M_ptr': empty_126, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_268 = triton_kernel_wrapper_mutation_230 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_14: "bf16[512, 512, 28, 28]" = torch.ops.aten.convolution.default(permute_31, convert_element_type_15, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  permute_31 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_59: "i64[]" = torch.ops.aten.add.Tensor(primals_87, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_344: "bf16[512, 512, 784]" = torch.ops.aten.reshape.default(convolution_14, [512, 512, 784]);  convolution_14 = None
        
        # No stacktrace found for following nodes
        as_strided_default_264: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_132: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_264);  as_strided_default_264 = None
        as_strided_default_265: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_132, [512], [1], 0);  clone_default_132 = None
        as_strided_default_266: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_133: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_266);  as_strided_default_266 = None
        as_strided_default_267: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_133, [512], [1], 0);  clone_default_133 = None
        triton_kernel_wrapper_mutation_229 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 275, constant_args_idx = 392, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_344, 'SUM': as_strided_default_265, 'SUMSQ': as_strided_default_267, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_229 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_42: "f32[512]" = torch.ops.aten.div.Tensor(as_strided_default_265, full_default_72);  as_strided_default_265 = None
        div_43: "f32[512]" = torch.ops.aten.div.Tensor(as_strided_default_267, full_default_72);  as_strided_default_267 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_228 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 276, constant_args_idx = 393, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_344, 'MEAN': div_42, 'INVSTD': rsqrt_14, 'GAMMA': primals_88, 'BETA': primals_89, 'Y': permute_40, 'X_hat': permute_41, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024});  view_344 = div_42 = primals_89 = triton_kernel_wrapper_mutation_228 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_129: "i32[401408, 32]" = torch.ops.aten.empty.memory_format([401408, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_130: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_131: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_348: "bf16[512, 784, 512]" = torch.ops.aten.reshape.default(permute_41, [512, -1, 512]);  permute_41 = None
        view_349: "bf16[401408, 512]" = torch.ops.aten.reshape.default(view_348, [401408, 512]);  view_348 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_227 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 277, constant_args_idx = 394, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_349, 'P_ptr': empty_129, 'S_ptr': empty_130, 'M_ptr': empty_131, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_349 = triton_kernel_wrapper_mutation_227 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:51 in forward, code: add_3 = layer2_0_bn3 + layer2_0_downsample_1;  layer2_0_bn3 = layer2_0_downsample_1 = None
        view_355: "bf16[512, 512, 28, 28]" = torch.ops.aten.reshape.default(permute_38, [512, 512, 28, 28]);  permute_38 = None
        view_356: "bf16[512, 512, 28, 28]" = torch.ops.aten.reshape.default(permute_40, [512, 512, 28, 28]);  permute_40 = None
        add_63: "bf16[512, 512, 28, 28]" = torch.ops.aten.add.Tensor(view_355, view_356);  view_355 = view_356 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_84: "i8[512, 512, 28, 28]" = torch.ops.aten.full.default([512, 512, 28, 28], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        empty_132: "bf16[512, 512, 28, 28]" = torch.ops.aten.empty.memory_format([512, 512, 28, 28], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_42: "bf16[512, 512, 28, 28]" = torch.ops.aten.permute.default(empty_132, [0, 1, 2, 3]);  empty_132 = None
        
        # No stacktrace found for following nodes
        as_strided_default_262: "i8[205520896]" = torch.ops.aten.as_strided.default(full_default_84, [205520896], [1], 0)
        clone_default_131: "i8[205520896]" = torch.ops.aten.clone.default(as_strided_default_262);  as_strided_default_262 = None
        as_strided_default_263: "i8[512, 512, 28, 28]" = torch.ops.aten.as_strided.default(clone_default_131, [512, 512, 28, 28], [401408, 784, 28, 1], 0);  clone_default_131 = None
        triton_kernel_wrapper_mutation_226 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 395, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_63, 'Y_ptr': permute_42, 'Mask_prt': as_strided_default_263, 'n_elts': 205520896, 'BLOCK_SIZE': 1024});  add_63 = triton_kernel_wrapper_mutation_226 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_359: "i8[512, 784, 512]" = torch.ops.aten.reshape.default(as_strided_default_263, [512, -1, 512]);  as_strided_default_263 = None
        view_360: "i8[401408, 512]" = torch.ops.aten.reshape.default(view_359, [401408, 512]);  view_359 = None
        
        # No stacktrace found for following nodes
        as_strided_default_260: "i32[6422528]" = torch.ops.aten.as_strided.default(full_default_69, [6422528], [1], 0)
        clone_default_130: "i32[6422528]" = torch.ops.aten.clone.default(as_strided_default_260);  as_strided_default_260 = None
        as_strided_default_261: "i32[401408, 16]" = torch.ops.aten.as_strided.default(clone_default_130, [401408, 16], [16, 1], 0);  clone_default_130 = None
        triton_kernel_wrapper_mutation_225 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 396, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_360, 'P_ptr': as_strided_default_261, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_360 = triton_kernel_wrapper_mutation_225 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_16: "bf16[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_92, torch.bfloat16);  primals_92 = None
        empty_133: "i32[401408, 32]" = torch.ops.aten.empty.memory_format([401408, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_134: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_135: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_363: "bf16[512, 784, 512]" = torch.ops.aten.reshape.default(permute_42, [512, -1, 512])
        view_364: "bf16[401408, 512]" = torch.ops.aten.reshape.default(view_363, [401408, 512]);  view_363 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_224 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 278, constant_args_idx = 397, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_364, 'P_ptr': empty_133, 'S_ptr': empty_134, 'M_ptr': empty_135, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_364 = triton_kernel_wrapper_mutation_224 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_15: "bf16[512, 128, 28, 28]" = torch.ops.aten.convolution.default(permute_42, convert_element_type_16, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_64: "i64[]" = torch.ops.aten.add.Tensor(primals_93, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_370: "bf16[512, 128, 784]" = torch.ops.aten.reshape.default(convolution_15, [512, 128, 784]);  convolution_15 = None
        
        # No stacktrace found for following nodes
        as_strided_default_256: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_128: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_256);  as_strided_default_256 = None
        as_strided_default_257: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_128, [128], [1], 0);  clone_default_128 = None
        as_strided_default_258: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_129: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_258);  as_strided_default_258 = None
        as_strided_default_259: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_129, [128], [1], 0);  clone_default_129 = None
        triton_kernel_wrapper_mutation_223 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 279, constant_args_idx = 398, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_370, 'SUM': as_strided_default_257, 'SUMSQ': as_strided_default_259, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_223 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_45: "f32[128]" = torch.ops.aten.div.Tensor(as_strided_default_257, full_default_72);  as_strided_default_257 = None
        div_46: "f32[128]" = torch.ops.aten.div.Tensor(as_strided_default_259, full_default_72);  as_strided_default_259 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_222 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 280, constant_args_idx = 399, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_370, 'MEAN': div_45, 'INVSTD': rsqrt_15, 'GAMMA': primals_94, 'BETA': primals_95, 'Y': permute_43, 'X_hat': permute_44, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024});  view_370 = div_45 = primals_95 = triton_kernel_wrapper_mutation_222 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_138: "i32[100352, 32]" = torch.ops.aten.empty.memory_format([100352, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_139: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_140: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_374: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(permute_44, [512, -1, 512]);  permute_44 = None
        view_375: "bf16[100352, 512]" = torch.ops.aten.reshape.default(view_374, [100352, 512]);  view_374 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_221 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 281, constant_args_idx = 400, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_375, 'P_ptr': empty_138, 'S_ptr': empty_139, 'M_ptr': empty_140, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_375 = triton_kernel_wrapper_mutation_221 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_381: "bf16[512, 128, 28, 28]" = torch.ops.aten.reshape.default(permute_43, [512, 128, 28, 28]);  permute_43 = None
        empty_141: "bf16[512, 128, 28, 28]" = torch.ops.aten.empty.memory_format([512, 128, 28, 28], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_45: "bf16[512, 128, 28, 28]" = torch.ops.aten.permute.default(empty_141, [0, 1, 2, 3]);  empty_141 = None
        
        # No stacktrace found for following nodes
        as_strided_default_254: "i8[51380224]" = torch.ops.aten.as_strided.default(full_default_74, [51380224], [1], 0)
        clone_default_127: "i8[51380224]" = torch.ops.aten.clone.default(as_strided_default_254);  as_strided_default_254 = None
        as_strided_default_255: "i8[512, 128, 28, 28]" = torch.ops.aten.as_strided.default(clone_default_127, [512, 128, 28, 28], [100352, 784, 28, 1], 0);  clone_default_127 = None
        triton_kernel_wrapper_mutation_220 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 401, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_381, 'Y_ptr': permute_45, 'Mask_prt': as_strided_default_255, 'n_elts': 51380224, 'BLOCK_SIZE': 1024});  view_381 = triton_kernel_wrapper_mutation_220 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_384: "i8[512, 196, 512]" = torch.ops.aten.reshape.default(as_strided_default_255, [512, -1, 512]);  as_strided_default_255 = None
        view_385: "i8[100352, 512]" = torch.ops.aten.reshape.default(view_384, [100352, 512]);  view_384 = None
        
        # No stacktrace found for following nodes
        as_strided_default_252: "i32[1605632]" = torch.ops.aten.as_strided.default(full_default_75, [1605632], [1], 0)
        clone_default_126: "i32[1605632]" = torch.ops.aten.clone.default(as_strided_default_252);  as_strided_default_252 = None
        as_strided_default_253: "i32[100352, 16]" = torch.ops.aten.as_strided.default(clone_default_126, [100352, 16], [16, 1], 0);  clone_default_126 = None
        triton_kernel_wrapper_mutation_219 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 402, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_385, 'P_ptr': as_strided_default_253, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_385 = triton_kernel_wrapper_mutation_219 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_17: "bf16[128, 128, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_98, torch.bfloat16);  primals_98 = None
        empty_142: "i32[100352, 32]" = torch.ops.aten.empty.memory_format([100352, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_143: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_144: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_388: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(permute_45, [512, -1, 512])
        view_389: "bf16[100352, 512]" = torch.ops.aten.reshape.default(view_388, [100352, 512]);  view_388 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_218 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 282, constant_args_idx = 403, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_389, 'P_ptr': empty_142, 'S_ptr': empty_143, 'M_ptr': empty_144, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_389 = triton_kernel_wrapper_mutation_218 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_16: "bf16[512, 128, 28, 28]" = torch.ops.aten.convolution.default(permute_45, convert_element_type_17, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  permute_45 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_68: "i64[]" = torch.ops.aten.add.Tensor(primals_99, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_395: "bf16[512, 128, 784]" = torch.ops.aten.reshape.default(convolution_16, [512, 128, 784]);  convolution_16 = None
        
        # No stacktrace found for following nodes
        as_strided_default_248: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_124: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_248);  as_strided_default_248 = None
        as_strided_default_249: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_124, [128], [1], 0);  clone_default_124 = None
        as_strided_default_250: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_125: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_250);  as_strided_default_250 = None
        as_strided_default_251: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_125, [128], [1], 0);  clone_default_125 = None
        triton_kernel_wrapper_mutation_217 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 283, constant_args_idx = 404, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_395, 'SUM': as_strided_default_249, 'SUMSQ': as_strided_default_251, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_217 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_48: "f32[128]" = torch.ops.aten.div.Tensor(as_strided_default_249, full_default_72);  as_strided_default_249 = None
        div_49: "f32[128]" = torch.ops.aten.div.Tensor(as_strided_default_251, full_default_72);  as_strided_default_251 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_216 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 284, constant_args_idx = 405, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_395, 'MEAN': div_48, 'INVSTD': rsqrt_16, 'GAMMA': primals_100, 'BETA': primals_101, 'Y': permute_46, 'X_hat': permute_47, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024});  view_395 = div_48 = primals_101 = triton_kernel_wrapper_mutation_216 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_147: "i32[100352, 32]" = torch.ops.aten.empty.memory_format([100352, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_148: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_149: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_399: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(permute_47, [512, -1, 512]);  permute_47 = None
        view_400: "bf16[100352, 512]" = torch.ops.aten.reshape.default(view_399, [100352, 512]);  view_399 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_215 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 285, constant_args_idx = 406, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_400, 'P_ptr': empty_147, 'S_ptr': empty_148, 'M_ptr': empty_149, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_400 = triton_kernel_wrapper_mutation_215 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_406: "bf16[512, 128, 28, 28]" = torch.ops.aten.reshape.default(permute_46, [512, 128, 28, 28]);  permute_46 = None
        empty_150: "bf16[512, 128, 28, 28]" = torch.ops.aten.empty.memory_format([512, 128, 28, 28], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_48: "bf16[512, 128, 28, 28]" = torch.ops.aten.permute.default(empty_150, [0, 1, 2, 3]);  empty_150 = None
        
        # No stacktrace found for following nodes
        as_strided_default_246: "i8[51380224]" = torch.ops.aten.as_strided.default(full_default_74, [51380224], [1], 0)
        clone_default_123: "i8[51380224]" = torch.ops.aten.clone.default(as_strided_default_246);  as_strided_default_246 = None
        as_strided_default_247: "i8[512, 128, 28, 28]" = torch.ops.aten.as_strided.default(clone_default_123, [512, 128, 28, 28], [100352, 784, 28, 1], 0);  clone_default_123 = None
        triton_kernel_wrapper_mutation_214 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 407, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_406, 'Y_ptr': permute_48, 'Mask_prt': as_strided_default_247, 'n_elts': 51380224, 'BLOCK_SIZE': 1024});  view_406 = triton_kernel_wrapper_mutation_214 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_409: "i8[512, 196, 512]" = torch.ops.aten.reshape.default(as_strided_default_247, [512, -1, 512]);  as_strided_default_247 = None
        view_410: "i8[100352, 512]" = torch.ops.aten.reshape.default(view_409, [100352, 512]);  view_409 = None
        
        # No stacktrace found for following nodes
        as_strided_default_244: "i32[1605632]" = torch.ops.aten.as_strided.default(full_default_75, [1605632], [1], 0)
        clone_default_122: "i32[1605632]" = torch.ops.aten.clone.default(as_strided_default_244);  as_strided_default_244 = None
        as_strided_default_245: "i32[100352, 16]" = torch.ops.aten.as_strided.default(clone_default_122, [100352, 16], [16, 1], 0);  clone_default_122 = None
        triton_kernel_wrapper_mutation_213 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 408, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_410, 'P_ptr': as_strided_default_245, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_410 = triton_kernel_wrapper_mutation_213 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_18: "bf16[512, 128, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_104, torch.bfloat16);  primals_104 = None
        empty_151: "i32[100352, 32]" = torch.ops.aten.empty.memory_format([100352, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_152: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_153: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_413: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(permute_48, [512, -1, 512])
        view_414: "bf16[100352, 512]" = torch.ops.aten.reshape.default(view_413, [100352, 512]);  view_413 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_212 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 286, constant_args_idx = 409, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_414, 'P_ptr': empty_151, 'S_ptr': empty_152, 'M_ptr': empty_153, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_414 = triton_kernel_wrapper_mutation_212 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_17: "bf16[512, 512, 28, 28]" = torch.ops.aten.convolution.default(permute_48, convert_element_type_18, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_48 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_72: "i64[]" = torch.ops.aten.add.Tensor(primals_105, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_420: "bf16[512, 512, 784]" = torch.ops.aten.reshape.default(convolution_17, [512, 512, 784]);  convolution_17 = None
        
        # No stacktrace found for following nodes
        as_strided_default_240: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_120: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_240);  as_strided_default_240 = None
        as_strided_default_241: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_120, [512], [1], 0);  clone_default_120 = None
        as_strided_default_242: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_121: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_242);  as_strided_default_242 = None
        as_strided_default_243: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_121, [512], [1], 0);  clone_default_121 = None
        triton_kernel_wrapper_mutation_211 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 287, constant_args_idx = 410, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_420, 'SUM': as_strided_default_241, 'SUMSQ': as_strided_default_243, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_211 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_51: "f32[512]" = torch.ops.aten.div.Tensor(as_strided_default_241, full_default_72);  as_strided_default_241 = None
        div_52: "f32[512]" = torch.ops.aten.div.Tensor(as_strided_default_243, full_default_72);  as_strided_default_243 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_210 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 288, constant_args_idx = 411, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_420, 'MEAN': div_51, 'INVSTD': rsqrt_17, 'GAMMA': primals_106, 'BETA': primals_107, 'Y': permute_49, 'X_hat': permute_50, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024});  view_420 = div_51 = primals_107 = triton_kernel_wrapper_mutation_210 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_156: "i32[401408, 32]" = torch.ops.aten.empty.memory_format([401408, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_157: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_158: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_424: "bf16[512, 784, 512]" = torch.ops.aten.reshape.default(permute_50, [512, -1, 512]);  permute_50 = None
        view_425: "bf16[401408, 512]" = torch.ops.aten.reshape.default(view_424, [401408, 512]);  view_424 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_209 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 289, constant_args_idx = 412, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_425, 'P_ptr': empty_156, 'S_ptr': empty_157, 'M_ptr': empty_158, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_425 = triton_kernel_wrapper_mutation_209 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:61 in forward, code: add_4 = layer2_1_bn3 + layer2_0_relu_2;  layer2_1_bn3 = layer2_0_relu_2 = None
        view_431: "bf16[512, 512, 28, 28]" = torch.ops.aten.reshape.default(permute_49, [512, 512, 28, 28]);  permute_49 = None
        add_76: "bf16[512, 512, 28, 28]" = torch.ops.aten.add.Tensor(view_431, permute_42);  view_431 = permute_42 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        empty_159: "bf16[512, 512, 28, 28]" = torch.ops.aten.empty.memory_format([512, 512, 28, 28], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_51: "bf16[512, 512, 28, 28]" = torch.ops.aten.permute.default(empty_159, [0, 1, 2, 3]);  empty_159 = None
        
        # No stacktrace found for following nodes
        as_strided_default_238: "i8[205520896]" = torch.ops.aten.as_strided.default(full_default_84, [205520896], [1], 0)
        clone_default_119: "i8[205520896]" = torch.ops.aten.clone.default(as_strided_default_238);  as_strided_default_238 = None
        as_strided_default_239: "i8[512, 512, 28, 28]" = torch.ops.aten.as_strided.default(clone_default_119, [512, 512, 28, 28], [401408, 784, 28, 1], 0);  clone_default_119 = None
        triton_kernel_wrapper_mutation_208 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 413, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_76, 'Y_ptr': permute_51, 'Mask_prt': as_strided_default_239, 'n_elts': 205520896, 'BLOCK_SIZE': 1024});  add_76 = triton_kernel_wrapper_mutation_208 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_434: "i8[512, 784, 512]" = torch.ops.aten.reshape.default(as_strided_default_239, [512, -1, 512]);  as_strided_default_239 = None
        view_435: "i8[401408, 512]" = torch.ops.aten.reshape.default(view_434, [401408, 512]);  view_434 = None
        
        # No stacktrace found for following nodes
        as_strided_default_236: "i32[6422528]" = torch.ops.aten.as_strided.default(full_default_69, [6422528], [1], 0)
        clone_default_118: "i32[6422528]" = torch.ops.aten.clone.default(as_strided_default_236);  as_strided_default_236 = None
        as_strided_default_237: "i32[401408, 16]" = torch.ops.aten.as_strided.default(clone_default_118, [401408, 16], [16, 1], 0);  clone_default_118 = None
        triton_kernel_wrapper_mutation_207 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 414, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_435, 'P_ptr': as_strided_default_237, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_435 = triton_kernel_wrapper_mutation_207 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_19: "bf16[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_110, torch.bfloat16);  primals_110 = None
        empty_160: "i32[401408, 32]" = torch.ops.aten.empty.memory_format([401408, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_161: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_162: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_438: "bf16[512, 784, 512]" = torch.ops.aten.reshape.default(permute_51, [512, -1, 512])
        view_439: "bf16[401408, 512]" = torch.ops.aten.reshape.default(view_438, [401408, 512]);  view_438 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_206 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 290, constant_args_idx = 415, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_439, 'P_ptr': empty_160, 'S_ptr': empty_161, 'M_ptr': empty_162, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_439 = triton_kernel_wrapper_mutation_206 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_18: "bf16[512, 128, 28, 28]" = torch.ops.aten.convolution.default(permute_51, convert_element_type_19, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_77: "i64[]" = torch.ops.aten.add.Tensor(primals_111, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_445: "bf16[512, 128, 784]" = torch.ops.aten.reshape.default(convolution_18, [512, 128, 784]);  convolution_18 = None
        
        # No stacktrace found for following nodes
        as_strided_default_232: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_116: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_232);  as_strided_default_232 = None
        as_strided_default_233: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_116, [128], [1], 0);  clone_default_116 = None
        as_strided_default_234: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_117: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_234);  as_strided_default_234 = None
        as_strided_default_235: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_117, [128], [1], 0);  clone_default_117 = None
        triton_kernel_wrapper_mutation_205 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 291, constant_args_idx = 416, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_445, 'SUM': as_strided_default_233, 'SUMSQ': as_strided_default_235, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_205 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_54: "f32[128]" = torch.ops.aten.div.Tensor(as_strided_default_233, full_default_72);  as_strided_default_233 = None
        div_55: "f32[128]" = torch.ops.aten.div.Tensor(as_strided_default_235, full_default_72);  as_strided_default_235 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_204 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 292, constant_args_idx = 417, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_445, 'MEAN': div_54, 'INVSTD': rsqrt_18, 'GAMMA': primals_112, 'BETA': primals_113, 'Y': permute_52, 'X_hat': permute_53, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024});  view_445 = div_54 = primals_113 = triton_kernel_wrapper_mutation_204 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_165: "i32[100352, 32]" = torch.ops.aten.empty.memory_format([100352, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_166: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_167: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_449: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(permute_53, [512, -1, 512]);  permute_53 = None
        view_450: "bf16[100352, 512]" = torch.ops.aten.reshape.default(view_449, [100352, 512]);  view_449 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_203 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 293, constant_args_idx = 418, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_450, 'P_ptr': empty_165, 'S_ptr': empty_166, 'M_ptr': empty_167, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_450 = triton_kernel_wrapper_mutation_203 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_456: "bf16[512, 128, 28, 28]" = torch.ops.aten.reshape.default(permute_52, [512, 128, 28, 28]);  permute_52 = None
        empty_168: "bf16[512, 128, 28, 28]" = torch.ops.aten.empty.memory_format([512, 128, 28, 28], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_54: "bf16[512, 128, 28, 28]" = torch.ops.aten.permute.default(empty_168, [0, 1, 2, 3]);  empty_168 = None
        
        # No stacktrace found for following nodes
        as_strided_default_230: "i8[51380224]" = torch.ops.aten.as_strided.default(full_default_74, [51380224], [1], 0)
        clone_default_115: "i8[51380224]" = torch.ops.aten.clone.default(as_strided_default_230);  as_strided_default_230 = None
        as_strided_default_231: "i8[512, 128, 28, 28]" = torch.ops.aten.as_strided.default(clone_default_115, [512, 128, 28, 28], [100352, 784, 28, 1], 0);  clone_default_115 = None
        triton_kernel_wrapper_mutation_202 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 419, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_456, 'Y_ptr': permute_54, 'Mask_prt': as_strided_default_231, 'n_elts': 51380224, 'BLOCK_SIZE': 1024});  view_456 = triton_kernel_wrapper_mutation_202 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_459: "i8[512, 196, 512]" = torch.ops.aten.reshape.default(as_strided_default_231, [512, -1, 512]);  as_strided_default_231 = None
        view_460: "i8[100352, 512]" = torch.ops.aten.reshape.default(view_459, [100352, 512]);  view_459 = None
        
        # No stacktrace found for following nodes
        as_strided_default_228: "i32[1605632]" = torch.ops.aten.as_strided.default(full_default_75, [1605632], [1], 0)
        clone_default_114: "i32[1605632]" = torch.ops.aten.clone.default(as_strided_default_228);  as_strided_default_228 = None
        as_strided_default_229: "i32[100352, 16]" = torch.ops.aten.as_strided.default(clone_default_114, [100352, 16], [16, 1], 0);  clone_default_114 = None
        triton_kernel_wrapper_mutation_201 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 420, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_460, 'P_ptr': as_strided_default_229, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_460 = triton_kernel_wrapper_mutation_201 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_20: "bf16[128, 128, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_116, torch.bfloat16);  primals_116 = None
        empty_169: "i32[100352, 32]" = torch.ops.aten.empty.memory_format([100352, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_170: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_171: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_463: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(permute_54, [512, -1, 512])
        view_464: "bf16[100352, 512]" = torch.ops.aten.reshape.default(view_463, [100352, 512]);  view_463 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_200 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 294, constant_args_idx = 421, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_464, 'P_ptr': empty_169, 'S_ptr': empty_170, 'M_ptr': empty_171, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_464 = triton_kernel_wrapper_mutation_200 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_19: "bf16[512, 128, 28, 28]" = torch.ops.aten.convolution.default(permute_54, convert_element_type_20, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  permute_54 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_81: "i64[]" = torch.ops.aten.add.Tensor(primals_117, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_470: "bf16[512, 128, 784]" = torch.ops.aten.reshape.default(convolution_19, [512, 128, 784]);  convolution_19 = None
        
        # No stacktrace found for following nodes
        as_strided_default_224: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_112: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_224);  as_strided_default_224 = None
        as_strided_default_225: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_112, [128], [1], 0);  clone_default_112 = None
        as_strided_default_226: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_113: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_226);  as_strided_default_226 = None
        as_strided_default_227: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_113, [128], [1], 0);  clone_default_113 = None
        triton_kernel_wrapper_mutation_199 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 295, constant_args_idx = 422, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_470, 'SUM': as_strided_default_225, 'SUMSQ': as_strided_default_227, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_199 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_57: "f32[128]" = torch.ops.aten.div.Tensor(as_strided_default_225, full_default_72);  as_strided_default_225 = None
        div_58: "f32[128]" = torch.ops.aten.div.Tensor(as_strided_default_227, full_default_72);  as_strided_default_227 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_198 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 296, constant_args_idx = 423, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_470, 'MEAN': div_57, 'INVSTD': rsqrt_19, 'GAMMA': primals_118, 'BETA': primals_119, 'Y': permute_55, 'X_hat': permute_56, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024});  view_470 = div_57 = primals_119 = triton_kernel_wrapper_mutation_198 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_174: "i32[100352, 32]" = torch.ops.aten.empty.memory_format([100352, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_175: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_176: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_474: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(permute_56, [512, -1, 512]);  permute_56 = None
        view_475: "bf16[100352, 512]" = torch.ops.aten.reshape.default(view_474, [100352, 512]);  view_474 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_197 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 297, constant_args_idx = 424, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_475, 'P_ptr': empty_174, 'S_ptr': empty_175, 'M_ptr': empty_176, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_475 = triton_kernel_wrapper_mutation_197 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_481: "bf16[512, 128, 28, 28]" = torch.ops.aten.reshape.default(permute_55, [512, 128, 28, 28]);  permute_55 = None
        empty_177: "bf16[512, 128, 28, 28]" = torch.ops.aten.empty.memory_format([512, 128, 28, 28], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_57: "bf16[512, 128, 28, 28]" = torch.ops.aten.permute.default(empty_177, [0, 1, 2, 3]);  empty_177 = None
        
        # No stacktrace found for following nodes
        as_strided_default_222: "i8[51380224]" = torch.ops.aten.as_strided.default(full_default_74, [51380224], [1], 0)
        clone_default_111: "i8[51380224]" = torch.ops.aten.clone.default(as_strided_default_222);  as_strided_default_222 = None
        as_strided_default_223: "i8[512, 128, 28, 28]" = torch.ops.aten.as_strided.default(clone_default_111, [512, 128, 28, 28], [100352, 784, 28, 1], 0);  clone_default_111 = None
        triton_kernel_wrapper_mutation_196 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 425, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_481, 'Y_ptr': permute_57, 'Mask_prt': as_strided_default_223, 'n_elts': 51380224, 'BLOCK_SIZE': 1024});  view_481 = triton_kernel_wrapper_mutation_196 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_484: "i8[512, 196, 512]" = torch.ops.aten.reshape.default(as_strided_default_223, [512, -1, 512]);  as_strided_default_223 = None
        view_485: "i8[100352, 512]" = torch.ops.aten.reshape.default(view_484, [100352, 512]);  view_484 = None
        
        # No stacktrace found for following nodes
        as_strided_default_220: "i32[1605632]" = torch.ops.aten.as_strided.default(full_default_75, [1605632], [1], 0)
        clone_default_110: "i32[1605632]" = torch.ops.aten.clone.default(as_strided_default_220);  as_strided_default_220 = None
        as_strided_default_221: "i32[100352, 16]" = torch.ops.aten.as_strided.default(clone_default_110, [100352, 16], [16, 1], 0);  clone_default_110 = None
        triton_kernel_wrapper_mutation_195 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 426, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_485, 'P_ptr': as_strided_default_221, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_485 = triton_kernel_wrapper_mutation_195 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_21: "bf16[512, 128, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_122, torch.bfloat16);  primals_122 = None
        empty_178: "i32[100352, 32]" = torch.ops.aten.empty.memory_format([100352, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_179: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_180: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_488: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(permute_57, [512, -1, 512])
        view_489: "bf16[100352, 512]" = torch.ops.aten.reshape.default(view_488, [100352, 512]);  view_488 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_194 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 298, constant_args_idx = 427, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_489, 'P_ptr': empty_178, 'S_ptr': empty_179, 'M_ptr': empty_180, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_489 = triton_kernel_wrapper_mutation_194 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_20: "bf16[512, 512, 28, 28]" = torch.ops.aten.convolution.default(permute_57, convert_element_type_21, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_57 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_85: "i64[]" = torch.ops.aten.add.Tensor(primals_123, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_495: "bf16[512, 512, 784]" = torch.ops.aten.reshape.default(convolution_20, [512, 512, 784]);  convolution_20 = None
        
        # No stacktrace found for following nodes
        as_strided_default_216: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_108: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_216);  as_strided_default_216 = None
        as_strided_default_217: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_108, [512], [1], 0);  clone_default_108 = None
        as_strided_default_218: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_109: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_218);  as_strided_default_218 = None
        as_strided_default_219: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_109, [512], [1], 0);  clone_default_109 = None
        triton_kernel_wrapper_mutation_193 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 299, constant_args_idx = 428, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_495, 'SUM': as_strided_default_217, 'SUMSQ': as_strided_default_219, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_193 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_60: "f32[512]" = torch.ops.aten.div.Tensor(as_strided_default_217, full_default_72);  as_strided_default_217 = None
        div_61: "f32[512]" = torch.ops.aten.div.Tensor(as_strided_default_219, full_default_72);  as_strided_default_219 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_192 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 300, constant_args_idx = 429, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_495, 'MEAN': div_60, 'INVSTD': rsqrt_20, 'GAMMA': primals_124, 'BETA': primals_125, 'Y': permute_58, 'X_hat': permute_59, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024});  view_495 = div_60 = primals_125 = triton_kernel_wrapper_mutation_192 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_183: "i32[401408, 32]" = torch.ops.aten.empty.memory_format([401408, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_184: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_185: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_499: "bf16[512, 784, 512]" = torch.ops.aten.reshape.default(permute_59, [512, -1, 512]);  permute_59 = None
        view_500: "bf16[401408, 512]" = torch.ops.aten.reshape.default(view_499, [401408, 512]);  view_499 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_191 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 301, constant_args_idx = 430, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_500, 'P_ptr': empty_183, 'S_ptr': empty_184, 'M_ptr': empty_185, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_500 = triton_kernel_wrapper_mutation_191 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:71 in forward, code: add_5 = layer2_2_bn3 + layer2_1_relu_2;  layer2_2_bn3 = layer2_1_relu_2 = None
        view_506: "bf16[512, 512, 28, 28]" = torch.ops.aten.reshape.default(permute_58, [512, 512, 28, 28]);  permute_58 = None
        add_89: "bf16[512, 512, 28, 28]" = torch.ops.aten.add.Tensor(view_506, permute_51);  view_506 = permute_51 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        empty_186: "bf16[512, 512, 28, 28]" = torch.ops.aten.empty.memory_format([512, 512, 28, 28], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_60: "bf16[512, 512, 28, 28]" = torch.ops.aten.permute.default(empty_186, [0, 1, 2, 3]);  empty_186 = None
        
        # No stacktrace found for following nodes
        as_strided_default_214: "i8[205520896]" = torch.ops.aten.as_strided.default(full_default_84, [205520896], [1], 0)
        clone_default_107: "i8[205520896]" = torch.ops.aten.clone.default(as_strided_default_214);  as_strided_default_214 = None
        as_strided_default_215: "i8[512, 512, 28, 28]" = torch.ops.aten.as_strided.default(clone_default_107, [512, 512, 28, 28], [401408, 784, 28, 1], 0);  clone_default_107 = None
        triton_kernel_wrapper_mutation_190 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 431, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_89, 'Y_ptr': permute_60, 'Mask_prt': as_strided_default_215, 'n_elts': 205520896, 'BLOCK_SIZE': 1024});  add_89 = triton_kernel_wrapper_mutation_190 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_509: "i8[512, 784, 512]" = torch.ops.aten.reshape.default(as_strided_default_215, [512, -1, 512]);  as_strided_default_215 = None
        view_510: "i8[401408, 512]" = torch.ops.aten.reshape.default(view_509, [401408, 512]);  view_509 = None
        
        # No stacktrace found for following nodes
        as_strided_default_212: "i32[6422528]" = torch.ops.aten.as_strided.default(full_default_69, [6422528], [1], 0)
        clone_default_106: "i32[6422528]" = torch.ops.aten.clone.default(as_strided_default_212);  as_strided_default_212 = None
        as_strided_default_213: "i32[401408, 16]" = torch.ops.aten.as_strided.default(clone_default_106, [401408, 16], [16, 1], 0);  clone_default_106 = None
        triton_kernel_wrapper_mutation_189 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 432, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_510, 'P_ptr': as_strided_default_213, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_510 = triton_kernel_wrapper_mutation_189 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_22: "bf16[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_128, torch.bfloat16);  primals_128 = None
        empty_187: "i32[401408, 32]" = torch.ops.aten.empty.memory_format([401408, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_188: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_189: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_513: "bf16[512, 784, 512]" = torch.ops.aten.reshape.default(permute_60, [512, -1, 512])
        view_514: "bf16[401408, 512]" = torch.ops.aten.reshape.default(view_513, [401408, 512]);  view_513 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_188 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 302, constant_args_idx = 433, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_514, 'P_ptr': empty_187, 'S_ptr': empty_188, 'M_ptr': empty_189, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_514 = triton_kernel_wrapper_mutation_188 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_21: "bf16[512, 128, 28, 28]" = torch.ops.aten.convolution.default(permute_60, convert_element_type_22, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_90: "i64[]" = torch.ops.aten.add.Tensor(primals_129, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_520: "bf16[512, 128, 784]" = torch.ops.aten.reshape.default(convolution_21, [512, 128, 784]);  convolution_21 = None
        
        # No stacktrace found for following nodes
        as_strided_default_208: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_104: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_208);  as_strided_default_208 = None
        as_strided_default_209: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_104, [128], [1], 0);  clone_default_104 = None
        as_strided_default_210: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_105: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_210);  as_strided_default_210 = None
        as_strided_default_211: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_105, [128], [1], 0);  clone_default_105 = None
        triton_kernel_wrapper_mutation_187 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 303, constant_args_idx = 434, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_520, 'SUM': as_strided_default_209, 'SUMSQ': as_strided_default_211, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_187 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_63: "f32[128]" = torch.ops.aten.div.Tensor(as_strided_default_209, full_default_72);  as_strided_default_209 = None
        div_64: "f32[128]" = torch.ops.aten.div.Tensor(as_strided_default_211, full_default_72);  as_strided_default_211 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_186 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 304, constant_args_idx = 435, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_520, 'MEAN': div_63, 'INVSTD': rsqrt_21, 'GAMMA': primals_130, 'BETA': primals_131, 'Y': permute_61, 'X_hat': permute_62, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024});  view_520 = div_63 = primals_131 = triton_kernel_wrapper_mutation_186 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_192: "i32[100352, 32]" = torch.ops.aten.empty.memory_format([100352, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_193: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_194: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_524: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(permute_62, [512, -1, 512]);  permute_62 = None
        view_525: "bf16[100352, 512]" = torch.ops.aten.reshape.default(view_524, [100352, 512]);  view_524 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_185 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 305, constant_args_idx = 436, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_525, 'P_ptr': empty_192, 'S_ptr': empty_193, 'M_ptr': empty_194, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_525 = triton_kernel_wrapper_mutation_185 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_531: "bf16[512, 128, 28, 28]" = torch.ops.aten.reshape.default(permute_61, [512, 128, 28, 28]);  permute_61 = None
        empty_195: "bf16[512, 128, 28, 28]" = torch.ops.aten.empty.memory_format([512, 128, 28, 28], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_63: "bf16[512, 128, 28, 28]" = torch.ops.aten.permute.default(empty_195, [0, 1, 2, 3]);  empty_195 = None
        
        # No stacktrace found for following nodes
        as_strided_default_206: "i8[51380224]" = torch.ops.aten.as_strided.default(full_default_74, [51380224], [1], 0)
        clone_default_103: "i8[51380224]" = torch.ops.aten.clone.default(as_strided_default_206);  as_strided_default_206 = None
        as_strided_default_207: "i8[512, 128, 28, 28]" = torch.ops.aten.as_strided.default(clone_default_103, [512, 128, 28, 28], [100352, 784, 28, 1], 0);  clone_default_103 = None
        triton_kernel_wrapper_mutation_184 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 437, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_531, 'Y_ptr': permute_63, 'Mask_prt': as_strided_default_207, 'n_elts': 51380224, 'BLOCK_SIZE': 1024});  view_531 = triton_kernel_wrapper_mutation_184 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_534: "i8[512, 196, 512]" = torch.ops.aten.reshape.default(as_strided_default_207, [512, -1, 512]);  as_strided_default_207 = None
        view_535: "i8[100352, 512]" = torch.ops.aten.reshape.default(view_534, [100352, 512]);  view_534 = None
        
        # No stacktrace found for following nodes
        as_strided_default_204: "i32[1605632]" = torch.ops.aten.as_strided.default(full_default_75, [1605632], [1], 0)
        clone_default_102: "i32[1605632]" = torch.ops.aten.clone.default(as_strided_default_204);  as_strided_default_204 = None
        as_strided_default_205: "i32[100352, 16]" = torch.ops.aten.as_strided.default(clone_default_102, [100352, 16], [16, 1], 0);  clone_default_102 = None
        triton_kernel_wrapper_mutation_183 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 438, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_535, 'P_ptr': as_strided_default_205, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_535 = triton_kernel_wrapper_mutation_183 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_23: "bf16[128, 128, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_134, torch.bfloat16);  primals_134 = None
        empty_196: "i32[100352, 32]" = torch.ops.aten.empty.memory_format([100352, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_197: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_198: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_538: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(permute_63, [512, -1, 512])
        view_539: "bf16[100352, 512]" = torch.ops.aten.reshape.default(view_538, [100352, 512]);  view_538 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_182 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 306, constant_args_idx = 439, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_539, 'P_ptr': empty_196, 'S_ptr': empty_197, 'M_ptr': empty_198, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_539 = triton_kernel_wrapper_mutation_182 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_22: "bf16[512, 128, 28, 28]" = torch.ops.aten.convolution.default(permute_63, convert_element_type_23, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  permute_63 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_94: "i64[]" = torch.ops.aten.add.Tensor(primals_135, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_545: "bf16[512, 128, 784]" = torch.ops.aten.reshape.default(convolution_22, [512, 128, 784]);  convolution_22 = None
        
        # No stacktrace found for following nodes
        as_strided_default_202: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_101: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_202);  as_strided_default_202 = None
        as_strided_default_203: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_101, [128], [1], 0);  clone_default_101 = None
        triton_kernel_wrapper_mutation_181 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 307, constant_args_idx = 440, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_545, 'SUM': full_default_64, 'SUMSQ': as_strided_default_203, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_181 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_66: "f32[128]" = torch.ops.aten.div.Tensor(full_default_64, full_default_72);  full_default_64 = None
        div_67: "f32[128]" = torch.ops.aten.div.Tensor(as_strided_default_203, full_default_72);  as_strided_default_203 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_180 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 308, constant_args_idx = 441, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_545, 'MEAN': div_66, 'INVSTD': rsqrt_22, 'GAMMA': primals_136, 'BETA': primals_137, 'Y': permute_64, 'X_hat': permute_65, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024});  view_545 = div_66 = primals_137 = triton_kernel_wrapper_mutation_180 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_201: "i32[100352, 32]" = torch.ops.aten.empty.memory_format([100352, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_202: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_203: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_549: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(permute_65, [512, -1, 512]);  permute_65 = None
        view_550: "bf16[100352, 512]" = torch.ops.aten.reshape.default(view_549, [100352, 512]);  view_549 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_179 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 309, constant_args_idx = 442, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_550, 'P_ptr': empty_201, 'S_ptr': empty_202, 'M_ptr': empty_203, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_550 = triton_kernel_wrapper_mutation_179 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_556: "bf16[512, 128, 28, 28]" = torch.ops.aten.reshape.default(permute_64, [512, 128, 28, 28]);  permute_64 = None
        empty_204: "bf16[512, 128, 28, 28]" = torch.ops.aten.empty.memory_format([512, 128, 28, 28], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_66: "bf16[512, 128, 28, 28]" = torch.ops.aten.permute.default(empty_204, [0, 1, 2, 3]);  empty_204 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_178 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 443, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_556, 'Y_ptr': permute_66, 'Mask_prt': full_default_74, 'n_elts': 51380224, 'BLOCK_SIZE': 1024});  view_556 = triton_kernel_wrapper_mutation_178 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_559: "i8[512, 196, 512]" = torch.ops.aten.reshape.default(full_default_74, [512, -1, 512]);  full_default_74 = None
        view_560: "i8[100352, 512]" = torch.ops.aten.reshape.default(view_559, [100352, 512]);  view_559 = None
        
        # No stacktrace found for following nodes
        as_strided_default_200: "i32[1605632]" = torch.ops.aten.as_strided.default(full_default_75, [1605632], [1], 0)
        clone_default_100: "i32[1605632]" = torch.ops.aten.clone.default(as_strided_default_200);  as_strided_default_200 = None
        as_strided_default_201: "i32[100352, 16]" = torch.ops.aten.as_strided.default(clone_default_100, [100352, 16], [16, 1], 0);  clone_default_100 = None
        triton_kernel_wrapper_mutation_177 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 444, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_560, 'P_ptr': as_strided_default_201, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_560 = triton_kernel_wrapper_mutation_177 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_24: "bf16[512, 128, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_140, torch.bfloat16);  primals_140 = None
        empty_205: "i32[100352, 32]" = torch.ops.aten.empty.memory_format([100352, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_206: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_207: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_563: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(permute_66, [512, -1, 512])
        view_564: "bf16[100352, 512]" = torch.ops.aten.reshape.default(view_563, [100352, 512]);  view_563 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_176 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 310, constant_args_idx = 445, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_564, 'P_ptr': empty_205, 'S_ptr': empty_206, 'M_ptr': empty_207, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_564 = triton_kernel_wrapper_mutation_176 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_23: "bf16[512, 512, 28, 28]" = torch.ops.aten.convolution.default(permute_66, convert_element_type_24, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_66 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_98: "i64[]" = torch.ops.aten.add.Tensor(primals_141, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_570: "bf16[512, 512, 784]" = torch.ops.aten.reshape.default(convolution_23, [512, 512, 784]);  convolution_23 = None
        
        # No stacktrace found for following nodes
        as_strided_default_196: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_98: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_196);  as_strided_default_196 = None
        as_strided_default_197: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_98, [512], [1], 0);  clone_default_98 = None
        as_strided_default_198: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_99: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_198);  as_strided_default_198 = None
        as_strided_default_199: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_99, [512], [1], 0);  clone_default_99 = None
        triton_kernel_wrapper_mutation_175 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 311, constant_args_idx = 446, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_570, 'SUM': as_strided_default_197, 'SUMSQ': as_strided_default_199, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_175 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_69: "f32[512]" = torch.ops.aten.div.Tensor(as_strided_default_197, full_default_72);  as_strided_default_197 = None
        div_70: "f32[512]" = torch.ops.aten.div.Tensor(as_strided_default_199, full_default_72);  as_strided_default_199 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_174 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 312, constant_args_idx = 447, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_570, 'MEAN': div_69, 'INVSTD': rsqrt_23, 'GAMMA': primals_142, 'BETA': primals_143, 'Y': permute_67, 'X_hat': permute_68, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024});  view_570 = div_69 = primals_143 = triton_kernel_wrapper_mutation_174 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_210: "i32[401408, 32]" = torch.ops.aten.empty.memory_format([401408, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_211: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_212: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_574: "bf16[512, 784, 512]" = torch.ops.aten.reshape.default(permute_68, [512, -1, 512]);  permute_68 = None
        view_575: "bf16[401408, 512]" = torch.ops.aten.reshape.default(view_574, [401408, 512]);  view_574 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_173 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 313, constant_args_idx = 448, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_575, 'P_ptr': empty_210, 'S_ptr': empty_211, 'M_ptr': empty_212, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_575 = triton_kernel_wrapper_mutation_173 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:81 in forward, code: add_6 = layer2_3_bn3 + layer2_2_relu_2;  layer2_3_bn3 = layer2_2_relu_2 = None
        view_581: "bf16[512, 512, 28, 28]" = torch.ops.aten.reshape.default(permute_67, [512, 512, 28, 28]);  permute_67 = None
        add_102: "bf16[512, 512, 28, 28]" = torch.ops.aten.add.Tensor(view_581, permute_60);  view_581 = permute_60 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        empty_213: "bf16[512, 512, 28, 28]" = torch.ops.aten.empty.memory_format([512, 512, 28, 28], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_69: "bf16[512, 512, 28, 28]" = torch.ops.aten.permute.default(empty_213, [0, 1, 2, 3]);  empty_213 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_172 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 449, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_102, 'Y_ptr': permute_69, 'Mask_prt': full_default_84, 'n_elts': 205520896, 'BLOCK_SIZE': 1024});  add_102 = triton_kernel_wrapper_mutation_172 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_584: "i8[512, 784, 512]" = torch.ops.aten.reshape.default(full_default_84, [512, -1, 512]);  full_default_84 = None
        view_585: "i8[401408, 512]" = torch.ops.aten.reshape.default(view_584, [401408, 512]);  view_584 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_171 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 450, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_585, 'P_ptr': full_default_69, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_585 = triton_kernel_wrapper_mutation_171 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_25: "bf16[256, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_146, torch.bfloat16);  primals_146 = None
        empty_214: "i32[401408, 32]" = torch.ops.aten.empty.memory_format([401408, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_215: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_216: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_588: "bf16[512, 784, 512]" = torch.ops.aten.reshape.default(permute_69, [512, -1, 512])
        view_589: "bf16[401408, 512]" = torch.ops.aten.reshape.default(view_588, [401408, 512]);  view_588 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_170 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 314, constant_args_idx = 451, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_589, 'P_ptr': empty_214, 'S_ptr': empty_215, 'M_ptr': empty_216, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  triton_kernel_wrapper_mutation_170 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_24: "bf16[512, 256, 28, 28]" = torch.ops.aten.convolution.default(permute_69, convert_element_type_25, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_103: "i64[]" = torch.ops.aten.add.Tensor(primals_147, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_595: "bf16[512, 256, 784]" = torch.ops.aten.reshape.default(convolution_24, [512, 256, 784]);  convolution_24 = None
        
        # No stacktrace found for following nodes
        as_strided_default_192: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_96: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_192);  as_strided_default_192 = None
        as_strided_default_193: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_96, [256], [1], 0);  clone_default_96 = None
        as_strided_default_194: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_97: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_194);  as_strided_default_194 = None
        as_strided_default_195: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_97, [256], [1], 0);  clone_default_97 = None
        triton_kernel_wrapper_mutation_169 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 315, constant_args_idx = 452, grid = [(256, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_595, 'SUM': as_strided_default_193, 'SUMSQ': as_strided_default_195, 'M': 401408, 'HW': 784, 'stride_n': 200704, 'stride_c': 784, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_169 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_72: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_193, full_default_72);  as_strided_default_193 = None
        div_73: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_195, full_default_72);  as_strided_default_195 = full_default_72 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_168 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 316, constant_args_idx = 453, grid = [(256, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_595, 'MEAN': div_72, 'INVSTD': rsqrt_24, 'GAMMA': primals_148, 'BETA': primals_149, 'Y': permute_70, 'X_hat': permute_71, 'M': 401408, 'HW': 784, 'stride_n': 200704, 'stride_c': 784, 'BLOCK_M': 1024});  view_595 = div_72 = primals_149 = triton_kernel_wrapper_mutation_168 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_219: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_220: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_221: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_599: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(permute_71, [512, -1, 512]);  permute_71 = None
        view_600: "bf16[200704, 512]" = torch.ops.aten.reshape.default(view_599, [200704, 512]);  view_599 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_167 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 317, constant_args_idx = 454, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_600, 'P_ptr': empty_219, 'S_ptr': empty_220, 'M_ptr': empty_221, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_600 = triton_kernel_wrapper_mutation_167 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_144: "i8[512, 256, 28, 28]" = torch.ops.aten.full.default([512, 256, 28, 28], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_606: "bf16[512, 256, 28, 28]" = torch.ops.aten.reshape.default(permute_70, [512, 256, 28, 28]);  permute_70 = None
        empty_222: "bf16[512, 256, 28, 28]" = torch.ops.aten.empty.memory_format([512, 256, 28, 28], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_72: "bf16[512, 256, 28, 28]" = torch.ops.aten.permute.default(empty_222, [0, 1, 2, 3]);  empty_222 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_166 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 455, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_606, 'Y_ptr': permute_72, 'Mask_prt': full_default_144, 'n_elts': 102760448, 'BLOCK_SIZE': 1024});  view_606 = triton_kernel_wrapper_mutation_166 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_609: "i8[512, 392, 512]" = torch.ops.aten.reshape.default(full_default_144, [512, -1, 512]);  full_default_144 = None
        view_610: "i8[200704, 512]" = torch.ops.aten.reshape.default(view_609, [200704, 512]);  view_609 = None
        
        # No stacktrace found for following nodes
        as_strided_default_190: "i32[3211264]" = torch.ops.aten.as_strided.default(full_default_11, [3211264], [1], 0)
        clone_default_95: "i32[3211264]" = torch.ops.aten.clone.default(as_strided_default_190);  as_strided_default_190 = None
        as_strided_default_191: "i32[200704, 16]" = torch.ops.aten.as_strided.default(clone_default_95, [200704, 16], [16, 1], 0);  clone_default_95 = None
        triton_kernel_wrapper_mutation_165 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 456, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_610, 'P_ptr': as_strided_default_191, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_610 = triton_kernel_wrapper_mutation_165 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_26: "bf16[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_152, torch.bfloat16);  primals_152 = None
        empty_223: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_224: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_225: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_613: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(permute_72, [512, -1, 512])
        view_614: "bf16[200704, 512]" = torch.ops.aten.reshape.default(view_613, [200704, 512]);  view_613 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_164 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 318, constant_args_idx = 457, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_614, 'P_ptr': empty_223, 'S_ptr': empty_224, 'M_ptr': empty_225, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_614 = triton_kernel_wrapper_mutation_164 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_25: "bf16[512, 256, 14, 14]" = torch.ops.aten.convolution.default(permute_72, convert_element_type_26, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  permute_72 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_107: "i64[]" = torch.ops.aten.add.Tensor(primals_153, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_620: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(convolution_25, [512, 256, 196]);  convolution_25 = None
        
        # No stacktrace found for following nodes
        as_strided_default_186: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_93: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_186);  as_strided_default_186 = None
        as_strided_default_187: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_93, [256], [1], 0);  clone_default_93 = None
        as_strided_default_188: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_94: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_188);  as_strided_default_188 = None
        as_strided_default_189: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_94, [256], [1], 0);  clone_default_94 = None
        triton_kernel_wrapper_mutation_163 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 319, constant_args_idx = 458, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_620, 'SUM': as_strided_default_187, 'SUMSQ': as_strided_default_189, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_163 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        full_default_148: "f32[]" = torch.ops.aten.full.default([], 100352.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        div_75: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_187, full_default_148);  as_strided_default_187 = None
        div_76: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_189, full_default_148);  as_strided_default_189 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_162 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 320, constant_args_idx = 459, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_620, 'MEAN': div_75, 'INVSTD': rsqrt_25, 'GAMMA': primals_154, 'BETA': primals_155, 'Y': permute_73, 'X_hat': permute_74, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  view_620 = div_75 = primals_155 = triton_kernel_wrapper_mutation_162 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_228: "i32[50176, 32]" = torch.ops.aten.empty.memory_format([50176, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_229: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_230: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_624: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(permute_74, [512, -1, 512]);  permute_74 = None
        view_625: "bf16[50176, 512]" = torch.ops.aten.reshape.default(view_624, [50176, 512]);  view_624 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_161 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 321, constant_args_idx = 460, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_625, 'P_ptr': empty_228, 'S_ptr': empty_229, 'M_ptr': empty_230, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_625 = triton_kernel_wrapper_mutation_161 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_150: "i8[512, 256, 14, 14]" = torch.ops.aten.full.default([512, 256, 14, 14], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_631: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(permute_73, [512, 256, 14, 14]);  permute_73 = None
        empty_231: "bf16[512, 256, 14, 14]" = torch.ops.aten.empty.memory_format([512, 256, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_75: "bf16[512, 256, 14, 14]" = torch.ops.aten.permute.default(empty_231, [0, 1, 2, 3]);  empty_231 = None
        
        # No stacktrace found for following nodes
        as_strided_default_184: "i8[25690112]" = torch.ops.aten.as_strided.default(full_default_150, [25690112], [1], 0)
        clone_default_92: "i8[25690112]" = torch.ops.aten.clone.default(as_strided_default_184);  as_strided_default_184 = None
        as_strided_default_185: "i8[512, 256, 14, 14]" = torch.ops.aten.as_strided.default(clone_default_92, [512, 256, 14, 14], [50176, 196, 14, 1], 0);  clone_default_92 = None
        triton_kernel_wrapper_mutation_160 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 461, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_631, 'Y_ptr': permute_75, 'Mask_prt': as_strided_default_185, 'n_elts': 25690112, 'BLOCK_SIZE': 1024});  view_631 = triton_kernel_wrapper_mutation_160 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_151: "i32[50176, 16]" = torch.ops.aten.full.default([50176, 16], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_634: "i8[512, 98, 512]" = torch.ops.aten.reshape.default(as_strided_default_185, [512, -1, 512]);  as_strided_default_185 = None
        view_635: "i8[50176, 512]" = torch.ops.aten.reshape.default(view_634, [50176, 512]);  view_634 = None
        
        # No stacktrace found for following nodes
        as_strided_default_182: "i32[802816]" = torch.ops.aten.as_strided.default(full_default_151, [802816], [1], 0)
        clone_default_91: "i32[802816]" = torch.ops.aten.clone.default(as_strided_default_182);  as_strided_default_182 = None
        as_strided_default_183: "i32[50176, 16]" = torch.ops.aten.as_strided.default(clone_default_91, [50176, 16], [16, 1], 0);  clone_default_91 = None
        triton_kernel_wrapper_mutation_159 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 462, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_635, 'P_ptr': as_strided_default_183, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_635 = triton_kernel_wrapper_mutation_159 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_27: "bf16[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_158, torch.bfloat16);  primals_158 = None
        empty_232: "i32[50176, 32]" = torch.ops.aten.empty.memory_format([50176, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_233: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_234: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_638: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(permute_75, [512, -1, 512])
        view_639: "bf16[50176, 512]" = torch.ops.aten.reshape.default(view_638, [50176, 512]);  view_638 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_158 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 322, constant_args_idx = 463, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_639, 'P_ptr': empty_232, 'S_ptr': empty_233, 'M_ptr': empty_234, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_639 = triton_kernel_wrapper_mutation_158 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_26: "bf16[512, 1024, 14, 14]" = torch.ops.aten.convolution.default(permute_75, convert_element_type_27, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_75 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_111: "i64[]" = torch.ops.aten.add.Tensor(primals_159, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_645: "bf16[512, 1024, 196]" = torch.ops.aten.reshape.default(convolution_26, [512, 1024, 196]);  convolution_26 = None
        full_default_152: "f32[1024]" = torch.ops.aten.full.default([1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        as_strided_default_178: "f32[1024]" = torch.ops.aten.as_strided.default(full_default_152, [1024], [1], 0)
        clone_default_89: "f32[1024]" = torch.ops.aten.clone.default(as_strided_default_178);  as_strided_default_178 = None
        as_strided_default_179: "f32[1024]" = torch.ops.aten.as_strided.default(clone_default_89, [1024], [1], 0);  clone_default_89 = None
        as_strided_default_180: "f32[1024]" = torch.ops.aten.as_strided.default(full_default_152, [1024], [1], 0)
        clone_default_90: "f32[1024]" = torch.ops.aten.clone.default(as_strided_default_180);  as_strided_default_180 = None
        as_strided_default_181: "f32[1024]" = torch.ops.aten.as_strided.default(clone_default_90, [1024], [1], 0);  clone_default_90 = None
        triton_kernel_wrapper_mutation_157 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 323, constant_args_idx = 464, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_645, 'SUM': as_strided_default_179, 'SUMSQ': as_strided_default_181, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_157 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_78: "f32[1024]" = torch.ops.aten.div.Tensor(as_strided_default_179, full_default_148);  as_strided_default_179 = None
        div_79: "f32[1024]" = torch.ops.aten.div.Tensor(as_strided_default_181, full_default_148);  as_strided_default_181 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_156 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 324, constant_args_idx = 465, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_645, 'MEAN': div_78, 'INVSTD': rsqrt_26, 'GAMMA': primals_160, 'BETA': primals_161, 'Y': permute_76, 'X_hat': permute_77, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024});  view_645 = div_78 = primals_161 = triton_kernel_wrapper_mutation_156 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_237: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_238: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_239: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_649: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(permute_77, [512, -1, 512]);  permute_77 = None
        view_650: "bf16[200704, 512]" = torch.ops.aten.reshape.default(view_649, [200704, 512]);  view_649 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_155 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 325, constant_args_idx = 466, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_650, 'P_ptr': empty_237, 'S_ptr': empty_238, 'M_ptr': empty_239, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_650 = triton_kernel_wrapper_mutation_155 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_28: "bf16[1024, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_164, torch.bfloat16);  primals_164 = None
        empty_240: "i32[401408, 32]" = torch.ops.aten.empty.memory_format([401408, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_241: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_242: "bf16[401408]" = torch.ops.aten.empty.memory_format([401408], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_154 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 326, constant_args_idx = 467, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_589, 'P_ptr': empty_240, 'S_ptr': empty_241, 'M_ptr': empty_242, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_589 = triton_kernel_wrapper_mutation_154 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_27: "bf16[512, 1024, 14, 14]" = torch.ops.aten.convolution.default(permute_69, convert_element_type_28, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  permute_69 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_115: "i64[]" = torch.ops.aten.add.Tensor(primals_165, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_665: "bf16[512, 1024, 196]" = torch.ops.aten.reshape.default(convolution_27, [512, 1024, 196]);  convolution_27 = None
        
        # No stacktrace found for following nodes
        as_strided_default_174: "f32[1024]" = torch.ops.aten.as_strided.default(full_default_152, [1024], [1], 0)
        clone_default_87: "f32[1024]" = torch.ops.aten.clone.default(as_strided_default_174);  as_strided_default_174 = None
        as_strided_default_175: "f32[1024]" = torch.ops.aten.as_strided.default(clone_default_87, [1024], [1], 0);  clone_default_87 = None
        as_strided_default_176: "f32[1024]" = torch.ops.aten.as_strided.default(full_default_152, [1024], [1], 0)
        clone_default_88: "f32[1024]" = torch.ops.aten.clone.default(as_strided_default_176);  as_strided_default_176 = None
        as_strided_default_177: "f32[1024]" = torch.ops.aten.as_strided.default(clone_default_88, [1024], [1], 0);  clone_default_88 = None
        triton_kernel_wrapper_mutation_153 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 327, constant_args_idx = 468, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_665, 'SUM': as_strided_default_175, 'SUMSQ': as_strided_default_177, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_153 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_81: "f32[1024]" = torch.ops.aten.div.Tensor(as_strided_default_175, full_default_148);  as_strided_default_175 = None
        div_82: "f32[1024]" = torch.ops.aten.div.Tensor(as_strided_default_177, full_default_148);  as_strided_default_177 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_152 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 328, constant_args_idx = 469, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_665, 'MEAN': div_81, 'INVSTD': rsqrt_27, 'GAMMA': primals_166, 'BETA': primals_167, 'Y': permute_78, 'X_hat': permute_79, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024});  view_665 = div_81 = primals_167 = triton_kernel_wrapper_mutation_152 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_245: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_246: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_247: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_669: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(permute_79, [512, -1, 512]);  permute_79 = None
        view_670: "bf16[200704, 512]" = torch.ops.aten.reshape.default(view_669, [200704, 512]);  view_669 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_151 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 329, constant_args_idx = 470, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_670, 'P_ptr': empty_245, 'S_ptr': empty_246, 'M_ptr': empty_247, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_670 = triton_kernel_wrapper_mutation_151 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:93 in forward, code: add_7 = layer3_0_bn3 + layer3_0_downsample_1;  layer3_0_bn3 = layer3_0_downsample_1 = None
        view_676: "bf16[512, 1024, 14, 14]" = torch.ops.aten.reshape.default(permute_76, [512, 1024, 14, 14]);  permute_76 = None
        view_677: "bf16[512, 1024, 14, 14]" = torch.ops.aten.reshape.default(permute_78, [512, 1024, 14, 14]);  permute_78 = None
        add_119: "bf16[512, 1024, 14, 14]" = torch.ops.aten.add.Tensor(view_676, view_677);  view_676 = view_677 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_160: "i8[512, 1024, 14, 14]" = torch.ops.aten.full.default([512, 1024, 14, 14], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        empty_248: "bf16[512, 1024, 14, 14]" = torch.ops.aten.empty.memory_format([512, 1024, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_80: "bf16[512, 1024, 14, 14]" = torch.ops.aten.permute.default(empty_248, [0, 1, 2, 3]);  empty_248 = None
        
        # No stacktrace found for following nodes
        as_strided_default_172: "i8[102760448]" = torch.ops.aten.as_strided.default(full_default_160, [102760448], [1], 0)
        clone_default_86: "i8[102760448]" = torch.ops.aten.clone.default(as_strided_default_172);  as_strided_default_172 = None
        as_strided_default_173: "i8[512, 1024, 14, 14]" = torch.ops.aten.as_strided.default(clone_default_86, [512, 1024, 14, 14], [200704, 196, 14, 1], 0);  clone_default_86 = None
        triton_kernel_wrapper_mutation_150 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 471, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_119, 'Y_ptr': permute_80, 'Mask_prt': as_strided_default_173, 'n_elts': 102760448, 'BLOCK_SIZE': 1024});  add_119 = triton_kernel_wrapper_mutation_150 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_680: "i8[512, 392, 512]" = torch.ops.aten.reshape.default(as_strided_default_173, [512, -1, 512]);  as_strided_default_173 = None
        view_681: "i8[200704, 512]" = torch.ops.aten.reshape.default(view_680, [200704, 512]);  view_680 = None
        
        # No stacktrace found for following nodes
        as_strided_default_170: "i32[3211264]" = torch.ops.aten.as_strided.default(full_default_11, [3211264], [1], 0)
        clone_default_85: "i32[3211264]" = torch.ops.aten.clone.default(as_strided_default_170);  as_strided_default_170 = None
        as_strided_default_171: "i32[200704, 16]" = torch.ops.aten.as_strided.default(clone_default_85, [200704, 16], [16, 1], 0);  clone_default_85 = None
        triton_kernel_wrapper_mutation_149 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 472, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_681, 'P_ptr': as_strided_default_171, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_681 = triton_kernel_wrapper_mutation_149 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_29: "bf16[256, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_170, torch.bfloat16);  primals_170 = None
        empty_249: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_250: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_251: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_684: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(permute_80, [512, -1, 512])
        view_685: "bf16[200704, 512]" = torch.ops.aten.reshape.default(view_684, [200704, 512]);  view_684 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_148 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 330, constant_args_idx = 473, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_685, 'P_ptr': empty_249, 'S_ptr': empty_250, 'M_ptr': empty_251, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_685 = triton_kernel_wrapper_mutation_148 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_28: "bf16[512, 256, 14, 14]" = torch.ops.aten.convolution.default(permute_80, convert_element_type_29, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_120: "i64[]" = torch.ops.aten.add.Tensor(primals_171, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_691: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(convolution_28, [512, 256, 196]);  convolution_28 = None
        
        # No stacktrace found for following nodes
        as_strided_default_166: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_83: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_166);  as_strided_default_166 = None
        as_strided_default_167: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_83, [256], [1], 0);  clone_default_83 = None
        as_strided_default_168: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_84: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_168);  as_strided_default_168 = None
        as_strided_default_169: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_84, [256], [1], 0);  clone_default_84 = None
        triton_kernel_wrapper_mutation_147 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 331, constant_args_idx = 474, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_691, 'SUM': as_strided_default_167, 'SUMSQ': as_strided_default_169, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_147 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_84: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_167, full_default_148);  as_strided_default_167 = None
        div_85: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_169, full_default_148);  as_strided_default_169 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_146 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 332, constant_args_idx = 475, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_691, 'MEAN': div_84, 'INVSTD': rsqrt_28, 'GAMMA': primals_172, 'BETA': primals_173, 'Y': permute_81, 'X_hat': permute_82, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  view_691 = div_84 = primals_173 = triton_kernel_wrapper_mutation_146 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_254: "i32[50176, 32]" = torch.ops.aten.empty.memory_format([50176, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_255: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_256: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_695: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(permute_82, [512, -1, 512]);  permute_82 = None
        view_696: "bf16[50176, 512]" = torch.ops.aten.reshape.default(view_695, [50176, 512]);  view_695 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_145 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 333, constant_args_idx = 476, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_696, 'P_ptr': empty_254, 'S_ptr': empty_255, 'M_ptr': empty_256, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_696 = triton_kernel_wrapper_mutation_145 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_702: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(permute_81, [512, 256, 14, 14]);  permute_81 = None
        empty_257: "bf16[512, 256, 14, 14]" = torch.ops.aten.empty.memory_format([512, 256, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_83: "bf16[512, 256, 14, 14]" = torch.ops.aten.permute.default(empty_257, [0, 1, 2, 3]);  empty_257 = None
        
        # No stacktrace found for following nodes
        as_strided_default_164: "i8[25690112]" = torch.ops.aten.as_strided.default(full_default_150, [25690112], [1], 0)
        clone_default_82: "i8[25690112]" = torch.ops.aten.clone.default(as_strided_default_164);  as_strided_default_164 = None
        as_strided_default_165: "i8[512, 256, 14, 14]" = torch.ops.aten.as_strided.default(clone_default_82, [512, 256, 14, 14], [50176, 196, 14, 1], 0);  clone_default_82 = None
        triton_kernel_wrapper_mutation_144 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 477, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_702, 'Y_ptr': permute_83, 'Mask_prt': as_strided_default_165, 'n_elts': 25690112, 'BLOCK_SIZE': 1024});  view_702 = triton_kernel_wrapper_mutation_144 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_705: "i8[512, 98, 512]" = torch.ops.aten.reshape.default(as_strided_default_165, [512, -1, 512]);  as_strided_default_165 = None
        view_706: "i8[50176, 512]" = torch.ops.aten.reshape.default(view_705, [50176, 512]);  view_705 = None
        
        # No stacktrace found for following nodes
        as_strided_default_162: "i32[802816]" = torch.ops.aten.as_strided.default(full_default_151, [802816], [1], 0)
        clone_default_81: "i32[802816]" = torch.ops.aten.clone.default(as_strided_default_162);  as_strided_default_162 = None
        as_strided_default_163: "i32[50176, 16]" = torch.ops.aten.as_strided.default(clone_default_81, [50176, 16], [16, 1], 0);  clone_default_81 = None
        triton_kernel_wrapper_mutation_143 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 478, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_706, 'P_ptr': as_strided_default_163, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_706 = triton_kernel_wrapper_mutation_143 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_30: "bf16[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_176, torch.bfloat16);  primals_176 = None
        empty_258: "i32[50176, 32]" = torch.ops.aten.empty.memory_format([50176, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_259: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_260: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_709: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(permute_83, [512, -1, 512])
        view_710: "bf16[50176, 512]" = torch.ops.aten.reshape.default(view_709, [50176, 512]);  view_709 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_142 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 334, constant_args_idx = 479, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_710, 'P_ptr': empty_258, 'S_ptr': empty_259, 'M_ptr': empty_260, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_710 = triton_kernel_wrapper_mutation_142 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_29: "bf16[512, 256, 14, 14]" = torch.ops.aten.convolution.default(permute_83, convert_element_type_30, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  permute_83 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_124: "i64[]" = torch.ops.aten.add.Tensor(primals_177, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_716: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(convolution_29, [512, 256, 196]);  convolution_29 = None
        
        # No stacktrace found for following nodes
        as_strided_default_158: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_79: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_158);  as_strided_default_158 = None
        as_strided_default_159: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_79, [256], [1], 0);  clone_default_79 = None
        as_strided_default_160: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_80: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_160);  as_strided_default_160 = None
        as_strided_default_161: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_80, [256], [1], 0);  clone_default_80 = None
        triton_kernel_wrapper_mutation_141 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 335, constant_args_idx = 480, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_716, 'SUM': as_strided_default_159, 'SUMSQ': as_strided_default_161, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_141 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_87: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_159, full_default_148);  as_strided_default_159 = None
        div_88: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_161, full_default_148);  as_strided_default_161 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_140 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 336, constant_args_idx = 481, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_716, 'MEAN': div_87, 'INVSTD': rsqrt_29, 'GAMMA': primals_178, 'BETA': primals_179, 'Y': permute_84, 'X_hat': permute_85, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  view_716 = div_87 = primals_179 = triton_kernel_wrapper_mutation_140 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_263: "i32[50176, 32]" = torch.ops.aten.empty.memory_format([50176, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_264: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_265: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_720: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(permute_85, [512, -1, 512]);  permute_85 = None
        view_721: "bf16[50176, 512]" = torch.ops.aten.reshape.default(view_720, [50176, 512]);  view_720 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_139 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 337, constant_args_idx = 482, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_721, 'P_ptr': empty_263, 'S_ptr': empty_264, 'M_ptr': empty_265, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_721 = triton_kernel_wrapper_mutation_139 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_727: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(permute_84, [512, 256, 14, 14]);  permute_84 = None
        empty_266: "bf16[512, 256, 14, 14]" = torch.ops.aten.empty.memory_format([512, 256, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_86: "bf16[512, 256, 14, 14]" = torch.ops.aten.permute.default(empty_266, [0, 1, 2, 3]);  empty_266 = None
        
        # No stacktrace found for following nodes
        as_strided_default_156: "i8[25690112]" = torch.ops.aten.as_strided.default(full_default_150, [25690112], [1], 0)
        clone_default_78: "i8[25690112]" = torch.ops.aten.clone.default(as_strided_default_156);  as_strided_default_156 = None
        as_strided_default_157: "i8[512, 256, 14, 14]" = torch.ops.aten.as_strided.default(clone_default_78, [512, 256, 14, 14], [50176, 196, 14, 1], 0);  clone_default_78 = None
        triton_kernel_wrapper_mutation_138 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 483, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_727, 'Y_ptr': permute_86, 'Mask_prt': as_strided_default_157, 'n_elts': 25690112, 'BLOCK_SIZE': 1024});  view_727 = triton_kernel_wrapper_mutation_138 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_730: "i8[512, 98, 512]" = torch.ops.aten.reshape.default(as_strided_default_157, [512, -1, 512]);  as_strided_default_157 = None
        view_731: "i8[50176, 512]" = torch.ops.aten.reshape.default(view_730, [50176, 512]);  view_730 = None
        
        # No stacktrace found for following nodes
        as_strided_default_154: "i32[802816]" = torch.ops.aten.as_strided.default(full_default_151, [802816], [1], 0)
        clone_default_77: "i32[802816]" = torch.ops.aten.clone.default(as_strided_default_154);  as_strided_default_154 = None
        as_strided_default_155: "i32[50176, 16]" = torch.ops.aten.as_strided.default(clone_default_77, [50176, 16], [16, 1], 0);  clone_default_77 = None
        triton_kernel_wrapper_mutation_137 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 484, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_731, 'P_ptr': as_strided_default_155, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_731 = triton_kernel_wrapper_mutation_137 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_31: "bf16[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_182, torch.bfloat16);  primals_182 = None
        empty_267: "i32[50176, 32]" = torch.ops.aten.empty.memory_format([50176, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_268: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_269: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_734: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(permute_86, [512, -1, 512])
        view_735: "bf16[50176, 512]" = torch.ops.aten.reshape.default(view_734, [50176, 512]);  view_734 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_136 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 338, constant_args_idx = 485, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_735, 'P_ptr': empty_267, 'S_ptr': empty_268, 'M_ptr': empty_269, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_735 = triton_kernel_wrapper_mutation_136 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_30: "bf16[512, 1024, 14, 14]" = torch.ops.aten.convolution.default(permute_86, convert_element_type_31, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_86 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_128: "i64[]" = torch.ops.aten.add.Tensor(primals_183, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_741: "bf16[512, 1024, 196]" = torch.ops.aten.reshape.default(convolution_30, [512, 1024, 196]);  convolution_30 = None
        
        # No stacktrace found for following nodes
        as_strided_default_150: "f32[1024]" = torch.ops.aten.as_strided.default(full_default_152, [1024], [1], 0)
        clone_default_75: "f32[1024]" = torch.ops.aten.clone.default(as_strided_default_150);  as_strided_default_150 = None
        as_strided_default_151: "f32[1024]" = torch.ops.aten.as_strided.default(clone_default_75, [1024], [1], 0);  clone_default_75 = None
        as_strided_default_152: "f32[1024]" = torch.ops.aten.as_strided.default(full_default_152, [1024], [1], 0)
        clone_default_76: "f32[1024]" = torch.ops.aten.clone.default(as_strided_default_152);  as_strided_default_152 = None
        as_strided_default_153: "f32[1024]" = torch.ops.aten.as_strided.default(clone_default_76, [1024], [1], 0);  clone_default_76 = None
        triton_kernel_wrapper_mutation_135 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 339, constant_args_idx = 486, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_741, 'SUM': as_strided_default_151, 'SUMSQ': as_strided_default_153, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_135 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_90: "f32[1024]" = torch.ops.aten.div.Tensor(as_strided_default_151, full_default_148);  as_strided_default_151 = None
        div_91: "f32[1024]" = torch.ops.aten.div.Tensor(as_strided_default_153, full_default_148);  as_strided_default_153 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_134 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 340, constant_args_idx = 487, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_741, 'MEAN': div_90, 'INVSTD': rsqrt_30, 'GAMMA': primals_184, 'BETA': primals_185, 'Y': permute_87, 'X_hat': permute_88, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024});  view_741 = div_90 = primals_185 = triton_kernel_wrapper_mutation_134 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_272: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_273: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_274: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_745: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(permute_88, [512, -1, 512]);  permute_88 = None
        view_746: "bf16[200704, 512]" = torch.ops.aten.reshape.default(view_745, [200704, 512]);  view_745 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_133 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 341, constant_args_idx = 488, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_746, 'P_ptr': empty_272, 'S_ptr': empty_273, 'M_ptr': empty_274, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_746 = triton_kernel_wrapper_mutation_133 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:103 in forward, code: add_8 = layer3_1_bn3 + layer3_0_relu_2;  layer3_1_bn3 = layer3_0_relu_2 = None
        view_752: "bf16[512, 1024, 14, 14]" = torch.ops.aten.reshape.default(permute_87, [512, 1024, 14, 14]);  permute_87 = None
        add_132: "bf16[512, 1024, 14, 14]" = torch.ops.aten.add.Tensor(view_752, permute_80);  view_752 = permute_80 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        empty_275: "bf16[512, 1024, 14, 14]" = torch.ops.aten.empty.memory_format([512, 1024, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_89: "bf16[512, 1024, 14, 14]" = torch.ops.aten.permute.default(empty_275, [0, 1, 2, 3]);  empty_275 = None
        
        # No stacktrace found for following nodes
        as_strided_default_148: "i8[102760448]" = torch.ops.aten.as_strided.default(full_default_160, [102760448], [1], 0)
        clone_default_74: "i8[102760448]" = torch.ops.aten.clone.default(as_strided_default_148);  as_strided_default_148 = None
        as_strided_default_149: "i8[512, 1024, 14, 14]" = torch.ops.aten.as_strided.default(clone_default_74, [512, 1024, 14, 14], [200704, 196, 14, 1], 0);  clone_default_74 = None
        triton_kernel_wrapper_mutation_132 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 489, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_132, 'Y_ptr': permute_89, 'Mask_prt': as_strided_default_149, 'n_elts': 102760448, 'BLOCK_SIZE': 1024});  add_132 = triton_kernel_wrapper_mutation_132 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_755: "i8[512, 392, 512]" = torch.ops.aten.reshape.default(as_strided_default_149, [512, -1, 512]);  as_strided_default_149 = None
        view_756: "i8[200704, 512]" = torch.ops.aten.reshape.default(view_755, [200704, 512]);  view_755 = None
        
        # No stacktrace found for following nodes
        as_strided_default_146: "i32[3211264]" = torch.ops.aten.as_strided.default(full_default_11, [3211264], [1], 0)
        clone_default_73: "i32[3211264]" = torch.ops.aten.clone.default(as_strided_default_146);  as_strided_default_146 = None
        as_strided_default_147: "i32[200704, 16]" = torch.ops.aten.as_strided.default(clone_default_73, [200704, 16], [16, 1], 0);  clone_default_73 = None
        triton_kernel_wrapper_mutation_131 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 490, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_756, 'P_ptr': as_strided_default_147, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_756 = triton_kernel_wrapper_mutation_131 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_32: "bf16[256, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_188, torch.bfloat16);  primals_188 = None
        empty_276: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_277: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_278: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_759: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(permute_89, [512, -1, 512])
        view_760: "bf16[200704, 512]" = torch.ops.aten.reshape.default(view_759, [200704, 512]);  view_759 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_130 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 342, constant_args_idx = 491, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_760, 'P_ptr': empty_276, 'S_ptr': empty_277, 'M_ptr': empty_278, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_760 = triton_kernel_wrapper_mutation_130 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_31: "bf16[512, 256, 14, 14]" = torch.ops.aten.convolution.default(permute_89, convert_element_type_32, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_133: "i64[]" = torch.ops.aten.add.Tensor(primals_189, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_766: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(convolution_31, [512, 256, 196]);  convolution_31 = None
        
        # No stacktrace found for following nodes
        as_strided_default_142: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_71: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_142);  as_strided_default_142 = None
        as_strided_default_143: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_71, [256], [1], 0);  clone_default_71 = None
        as_strided_default_144: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_72: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_144);  as_strided_default_144 = None
        as_strided_default_145: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_72, [256], [1], 0);  clone_default_72 = None
        triton_kernel_wrapper_mutation_129 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 343, constant_args_idx = 492, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_766, 'SUM': as_strided_default_143, 'SUMSQ': as_strided_default_145, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_129 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_93: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_143, full_default_148);  as_strided_default_143 = None
        div_94: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_145, full_default_148);  as_strided_default_145 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_128 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 344, constant_args_idx = 493, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_766, 'MEAN': div_93, 'INVSTD': rsqrt_31, 'GAMMA': primals_190, 'BETA': primals_191, 'Y': permute_90, 'X_hat': permute_91, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  view_766 = div_93 = primals_191 = triton_kernel_wrapper_mutation_128 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_281: "i32[50176, 32]" = torch.ops.aten.empty.memory_format([50176, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_282: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_283: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_770: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(permute_91, [512, -1, 512]);  permute_91 = None
        view_771: "bf16[50176, 512]" = torch.ops.aten.reshape.default(view_770, [50176, 512]);  view_770 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_127 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 345, constant_args_idx = 494, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_771, 'P_ptr': empty_281, 'S_ptr': empty_282, 'M_ptr': empty_283, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_771 = triton_kernel_wrapper_mutation_127 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_777: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(permute_90, [512, 256, 14, 14]);  permute_90 = None
        empty_284: "bf16[512, 256, 14, 14]" = torch.ops.aten.empty.memory_format([512, 256, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_92: "bf16[512, 256, 14, 14]" = torch.ops.aten.permute.default(empty_284, [0, 1, 2, 3]);  empty_284 = None
        
        # No stacktrace found for following nodes
        as_strided_default_140: "i8[25690112]" = torch.ops.aten.as_strided.default(full_default_150, [25690112], [1], 0)
        clone_default_70: "i8[25690112]" = torch.ops.aten.clone.default(as_strided_default_140);  as_strided_default_140 = None
        as_strided_default_141: "i8[512, 256, 14, 14]" = torch.ops.aten.as_strided.default(clone_default_70, [512, 256, 14, 14], [50176, 196, 14, 1], 0);  clone_default_70 = None
        triton_kernel_wrapper_mutation_126 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 495, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_777, 'Y_ptr': permute_92, 'Mask_prt': as_strided_default_141, 'n_elts': 25690112, 'BLOCK_SIZE': 1024});  view_777 = triton_kernel_wrapper_mutation_126 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_780: "i8[512, 98, 512]" = torch.ops.aten.reshape.default(as_strided_default_141, [512, -1, 512]);  as_strided_default_141 = None
        view_781: "i8[50176, 512]" = torch.ops.aten.reshape.default(view_780, [50176, 512]);  view_780 = None
        
        # No stacktrace found for following nodes
        as_strided_default_138: "i32[802816]" = torch.ops.aten.as_strided.default(full_default_151, [802816], [1], 0)
        clone_default_69: "i32[802816]" = torch.ops.aten.clone.default(as_strided_default_138);  as_strided_default_138 = None
        as_strided_default_139: "i32[50176, 16]" = torch.ops.aten.as_strided.default(clone_default_69, [50176, 16], [16, 1], 0);  clone_default_69 = None
        triton_kernel_wrapper_mutation_125 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 496, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_781, 'P_ptr': as_strided_default_139, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_781 = triton_kernel_wrapper_mutation_125 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_33: "bf16[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_194, torch.bfloat16);  primals_194 = None
        empty_285: "i32[50176, 32]" = torch.ops.aten.empty.memory_format([50176, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_286: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_287: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_784: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(permute_92, [512, -1, 512])
        view_785: "bf16[50176, 512]" = torch.ops.aten.reshape.default(view_784, [50176, 512]);  view_784 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_124 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 346, constant_args_idx = 497, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_785, 'P_ptr': empty_285, 'S_ptr': empty_286, 'M_ptr': empty_287, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_785 = triton_kernel_wrapper_mutation_124 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_32: "bf16[512, 256, 14, 14]" = torch.ops.aten.convolution.default(permute_92, convert_element_type_33, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  permute_92 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_137: "i64[]" = torch.ops.aten.add.Tensor(primals_195, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_791: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(convolution_32, [512, 256, 196]);  convolution_32 = None
        
        # No stacktrace found for following nodes
        as_strided_default_134: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_67: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_134);  as_strided_default_134 = None
        as_strided_default_135: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_67, [256], [1], 0);  clone_default_67 = None
        as_strided_default_136: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_68: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_136);  as_strided_default_136 = None
        as_strided_default_137: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_68, [256], [1], 0);  clone_default_68 = None
        triton_kernel_wrapper_mutation_123 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 347, constant_args_idx = 498, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_791, 'SUM': as_strided_default_135, 'SUMSQ': as_strided_default_137, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_123 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_96: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_135, full_default_148);  as_strided_default_135 = None
        div_97: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_137, full_default_148);  as_strided_default_137 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_122 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 348, constant_args_idx = 499, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_791, 'MEAN': div_96, 'INVSTD': rsqrt_32, 'GAMMA': primals_196, 'BETA': primals_197, 'Y': permute_93, 'X_hat': permute_94, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  view_791 = div_96 = primals_197 = triton_kernel_wrapper_mutation_122 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_290: "i32[50176, 32]" = torch.ops.aten.empty.memory_format([50176, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_291: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_292: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_795: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(permute_94, [512, -1, 512]);  permute_94 = None
        view_796: "bf16[50176, 512]" = torch.ops.aten.reshape.default(view_795, [50176, 512]);  view_795 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_121 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 349, constant_args_idx = 500, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_796, 'P_ptr': empty_290, 'S_ptr': empty_291, 'M_ptr': empty_292, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_796 = triton_kernel_wrapper_mutation_121 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_802: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(permute_93, [512, 256, 14, 14]);  permute_93 = None
        empty_293: "bf16[512, 256, 14, 14]" = torch.ops.aten.empty.memory_format([512, 256, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_95: "bf16[512, 256, 14, 14]" = torch.ops.aten.permute.default(empty_293, [0, 1, 2, 3]);  empty_293 = None
        
        # No stacktrace found for following nodes
        as_strided_default_132: "i8[25690112]" = torch.ops.aten.as_strided.default(full_default_150, [25690112], [1], 0)
        clone_default_66: "i8[25690112]" = torch.ops.aten.clone.default(as_strided_default_132);  as_strided_default_132 = None
        as_strided_default_133: "i8[512, 256, 14, 14]" = torch.ops.aten.as_strided.default(clone_default_66, [512, 256, 14, 14], [50176, 196, 14, 1], 0);  clone_default_66 = None
        triton_kernel_wrapper_mutation_120 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 501, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_802, 'Y_ptr': permute_95, 'Mask_prt': as_strided_default_133, 'n_elts': 25690112, 'BLOCK_SIZE': 1024});  view_802 = triton_kernel_wrapper_mutation_120 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_805: "i8[512, 98, 512]" = torch.ops.aten.reshape.default(as_strided_default_133, [512, -1, 512]);  as_strided_default_133 = None
        view_806: "i8[50176, 512]" = torch.ops.aten.reshape.default(view_805, [50176, 512]);  view_805 = None
        
        # No stacktrace found for following nodes
        as_strided_default_130: "i32[802816]" = torch.ops.aten.as_strided.default(full_default_151, [802816], [1], 0)
        clone_default_65: "i32[802816]" = torch.ops.aten.clone.default(as_strided_default_130);  as_strided_default_130 = None
        as_strided_default_131: "i32[50176, 16]" = torch.ops.aten.as_strided.default(clone_default_65, [50176, 16], [16, 1], 0);  clone_default_65 = None
        triton_kernel_wrapper_mutation_119 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 502, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_806, 'P_ptr': as_strided_default_131, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_806 = triton_kernel_wrapper_mutation_119 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_34: "bf16[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_200, torch.bfloat16);  primals_200 = None
        empty_294: "i32[50176, 32]" = torch.ops.aten.empty.memory_format([50176, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_295: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_296: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_809: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(permute_95, [512, -1, 512])
        view_810: "bf16[50176, 512]" = torch.ops.aten.reshape.default(view_809, [50176, 512]);  view_809 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_118 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 350, constant_args_idx = 503, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_810, 'P_ptr': empty_294, 'S_ptr': empty_295, 'M_ptr': empty_296, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_810 = triton_kernel_wrapper_mutation_118 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_33: "bf16[512, 1024, 14, 14]" = torch.ops.aten.convolution.default(permute_95, convert_element_type_34, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_95 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_141: "i64[]" = torch.ops.aten.add.Tensor(primals_201, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_816: "bf16[512, 1024, 196]" = torch.ops.aten.reshape.default(convolution_33, [512, 1024, 196]);  convolution_33 = None
        
        # No stacktrace found for following nodes
        as_strided_default_126: "f32[1024]" = torch.ops.aten.as_strided.default(full_default_152, [1024], [1], 0)
        clone_default_63: "f32[1024]" = torch.ops.aten.clone.default(as_strided_default_126);  as_strided_default_126 = None
        as_strided_default_127: "f32[1024]" = torch.ops.aten.as_strided.default(clone_default_63, [1024], [1], 0);  clone_default_63 = None
        as_strided_default_128: "f32[1024]" = torch.ops.aten.as_strided.default(full_default_152, [1024], [1], 0)
        clone_default_64: "f32[1024]" = torch.ops.aten.clone.default(as_strided_default_128);  as_strided_default_128 = None
        as_strided_default_129: "f32[1024]" = torch.ops.aten.as_strided.default(clone_default_64, [1024], [1], 0);  clone_default_64 = None
        triton_kernel_wrapper_mutation_117 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 351, constant_args_idx = 504, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_816, 'SUM': as_strided_default_127, 'SUMSQ': as_strided_default_129, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_117 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_99: "f32[1024]" = torch.ops.aten.div.Tensor(as_strided_default_127, full_default_148);  as_strided_default_127 = None
        div_100: "f32[1024]" = torch.ops.aten.div.Tensor(as_strided_default_129, full_default_148);  as_strided_default_129 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_116 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 352, constant_args_idx = 505, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_816, 'MEAN': div_99, 'INVSTD': rsqrt_33, 'GAMMA': primals_202, 'BETA': primals_203, 'Y': permute_96, 'X_hat': permute_97, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024});  view_816 = div_99 = primals_203 = triton_kernel_wrapper_mutation_116 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_299: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_300: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_301: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_820: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(permute_97, [512, -1, 512]);  permute_97 = None
        view_821: "bf16[200704, 512]" = torch.ops.aten.reshape.default(view_820, [200704, 512]);  view_820 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_115 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 353, constant_args_idx = 506, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_821, 'P_ptr': empty_299, 'S_ptr': empty_300, 'M_ptr': empty_301, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_821 = triton_kernel_wrapper_mutation_115 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:113 in forward, code: add_9 = layer3_2_bn3 + layer3_1_relu_2;  layer3_2_bn3 = layer3_1_relu_2 = None
        view_827: "bf16[512, 1024, 14, 14]" = torch.ops.aten.reshape.default(permute_96, [512, 1024, 14, 14]);  permute_96 = None
        add_145: "bf16[512, 1024, 14, 14]" = torch.ops.aten.add.Tensor(view_827, permute_89);  view_827 = permute_89 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        empty_302: "bf16[512, 1024, 14, 14]" = torch.ops.aten.empty.memory_format([512, 1024, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_98: "bf16[512, 1024, 14, 14]" = torch.ops.aten.permute.default(empty_302, [0, 1, 2, 3]);  empty_302 = None
        
        # No stacktrace found for following nodes
        as_strided_default_124: "i8[102760448]" = torch.ops.aten.as_strided.default(full_default_160, [102760448], [1], 0)
        clone_default_62: "i8[102760448]" = torch.ops.aten.clone.default(as_strided_default_124);  as_strided_default_124 = None
        as_strided_default_125: "i8[512, 1024, 14, 14]" = torch.ops.aten.as_strided.default(clone_default_62, [512, 1024, 14, 14], [200704, 196, 14, 1], 0);  clone_default_62 = None
        triton_kernel_wrapper_mutation_114 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 507, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_145, 'Y_ptr': permute_98, 'Mask_prt': as_strided_default_125, 'n_elts': 102760448, 'BLOCK_SIZE': 1024});  add_145 = triton_kernel_wrapper_mutation_114 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_830: "i8[512, 392, 512]" = torch.ops.aten.reshape.default(as_strided_default_125, [512, -1, 512]);  as_strided_default_125 = None
        view_831: "i8[200704, 512]" = torch.ops.aten.reshape.default(view_830, [200704, 512]);  view_830 = None
        
        # No stacktrace found for following nodes
        as_strided_default_122: "i32[3211264]" = torch.ops.aten.as_strided.default(full_default_11, [3211264], [1], 0)
        clone_default_61: "i32[3211264]" = torch.ops.aten.clone.default(as_strided_default_122);  as_strided_default_122 = None
        as_strided_default_123: "i32[200704, 16]" = torch.ops.aten.as_strided.default(clone_default_61, [200704, 16], [16, 1], 0);  clone_default_61 = None
        triton_kernel_wrapper_mutation_113 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 508, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_831, 'P_ptr': as_strided_default_123, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_831 = triton_kernel_wrapper_mutation_113 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_35: "bf16[256, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_206, torch.bfloat16);  primals_206 = None
        empty_303: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_304: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_305: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_834: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(permute_98, [512, -1, 512])
        view_835: "bf16[200704, 512]" = torch.ops.aten.reshape.default(view_834, [200704, 512]);  view_834 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_112 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 354, constant_args_idx = 509, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_835, 'P_ptr': empty_303, 'S_ptr': empty_304, 'M_ptr': empty_305, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_835 = triton_kernel_wrapper_mutation_112 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_34: "bf16[512, 256, 14, 14]" = torch.ops.aten.convolution.default(permute_98, convert_element_type_35, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_146: "i64[]" = torch.ops.aten.add.Tensor(primals_207, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_841: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(convolution_34, [512, 256, 196]);  convolution_34 = None
        
        # No stacktrace found for following nodes
        as_strided_default_118: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_59: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_118);  as_strided_default_118 = None
        as_strided_default_119: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_59, [256], [1], 0);  clone_default_59 = None
        as_strided_default_120: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_60: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_120);  as_strided_default_120 = None
        as_strided_default_121: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_60, [256], [1], 0);  clone_default_60 = None
        triton_kernel_wrapper_mutation_111 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 355, constant_args_idx = 510, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_841, 'SUM': as_strided_default_119, 'SUMSQ': as_strided_default_121, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_111 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_102: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_119, full_default_148);  as_strided_default_119 = None
        div_103: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_121, full_default_148);  as_strided_default_121 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_110 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 356, constant_args_idx = 511, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_841, 'MEAN': div_102, 'INVSTD': rsqrt_34, 'GAMMA': primals_208, 'BETA': primals_209, 'Y': permute_99, 'X_hat': permute_100, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  view_841 = div_102 = primals_209 = triton_kernel_wrapper_mutation_110 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_308: "i32[50176, 32]" = torch.ops.aten.empty.memory_format([50176, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_309: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_310: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_845: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(permute_100, [512, -1, 512]);  permute_100 = None
        view_846: "bf16[50176, 512]" = torch.ops.aten.reshape.default(view_845, [50176, 512]);  view_845 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_109 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 357, constant_args_idx = 512, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_846, 'P_ptr': empty_308, 'S_ptr': empty_309, 'M_ptr': empty_310, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_846 = triton_kernel_wrapper_mutation_109 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_852: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(permute_99, [512, 256, 14, 14]);  permute_99 = None
        empty_311: "bf16[512, 256, 14, 14]" = torch.ops.aten.empty.memory_format([512, 256, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_101: "bf16[512, 256, 14, 14]" = torch.ops.aten.permute.default(empty_311, [0, 1, 2, 3]);  empty_311 = None
        
        # No stacktrace found for following nodes
        as_strided_default_116: "i8[25690112]" = torch.ops.aten.as_strided.default(full_default_150, [25690112], [1], 0)
        clone_default_58: "i8[25690112]" = torch.ops.aten.clone.default(as_strided_default_116);  as_strided_default_116 = None
        as_strided_default_117: "i8[512, 256, 14, 14]" = torch.ops.aten.as_strided.default(clone_default_58, [512, 256, 14, 14], [50176, 196, 14, 1], 0);  clone_default_58 = None
        triton_kernel_wrapper_mutation_108 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 513, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_852, 'Y_ptr': permute_101, 'Mask_prt': as_strided_default_117, 'n_elts': 25690112, 'BLOCK_SIZE': 1024});  view_852 = triton_kernel_wrapper_mutation_108 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_855: "i8[512, 98, 512]" = torch.ops.aten.reshape.default(as_strided_default_117, [512, -1, 512]);  as_strided_default_117 = None
        view_856: "i8[50176, 512]" = torch.ops.aten.reshape.default(view_855, [50176, 512]);  view_855 = None
        
        # No stacktrace found for following nodes
        as_strided_default_114: "i32[802816]" = torch.ops.aten.as_strided.default(full_default_151, [802816], [1], 0)
        clone_default_57: "i32[802816]" = torch.ops.aten.clone.default(as_strided_default_114);  as_strided_default_114 = None
        as_strided_default_115: "i32[50176, 16]" = torch.ops.aten.as_strided.default(clone_default_57, [50176, 16], [16, 1], 0);  clone_default_57 = None
        triton_kernel_wrapper_mutation_107 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 514, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_856, 'P_ptr': as_strided_default_115, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_856 = triton_kernel_wrapper_mutation_107 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_36: "bf16[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_212, torch.bfloat16);  primals_212 = None
        empty_312: "i32[50176, 32]" = torch.ops.aten.empty.memory_format([50176, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_313: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_314: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_859: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(permute_101, [512, -1, 512])
        view_860: "bf16[50176, 512]" = torch.ops.aten.reshape.default(view_859, [50176, 512]);  view_859 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_106 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 358, constant_args_idx = 515, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_860, 'P_ptr': empty_312, 'S_ptr': empty_313, 'M_ptr': empty_314, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_860 = triton_kernel_wrapper_mutation_106 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_35: "bf16[512, 256, 14, 14]" = torch.ops.aten.convolution.default(permute_101, convert_element_type_36, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  permute_101 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_150: "i64[]" = torch.ops.aten.add.Tensor(primals_213, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_866: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(convolution_35, [512, 256, 196]);  convolution_35 = None
        
        # No stacktrace found for following nodes
        as_strided_default_110: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_55: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_110);  as_strided_default_110 = None
        as_strided_default_111: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_55, [256], [1], 0);  clone_default_55 = None
        as_strided_default_112: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_56: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_112);  as_strided_default_112 = None
        as_strided_default_113: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_56, [256], [1], 0);  clone_default_56 = None
        triton_kernel_wrapper_mutation_105 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 359, constant_args_idx = 516, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_866, 'SUM': as_strided_default_111, 'SUMSQ': as_strided_default_113, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_105 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_105: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_111, full_default_148);  as_strided_default_111 = None
        div_106: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_113, full_default_148);  as_strided_default_113 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_104 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 360, constant_args_idx = 517, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_866, 'MEAN': div_105, 'INVSTD': rsqrt_35, 'GAMMA': primals_214, 'BETA': primals_215, 'Y': permute_102, 'X_hat': permute_103, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  view_866 = div_105 = primals_215 = triton_kernel_wrapper_mutation_104 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_317: "i32[50176, 32]" = torch.ops.aten.empty.memory_format([50176, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_318: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_319: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_870: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(permute_103, [512, -1, 512]);  permute_103 = None
        view_871: "bf16[50176, 512]" = torch.ops.aten.reshape.default(view_870, [50176, 512]);  view_870 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_103 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 361, constant_args_idx = 518, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_871, 'P_ptr': empty_317, 'S_ptr': empty_318, 'M_ptr': empty_319, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_871 = triton_kernel_wrapper_mutation_103 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_877: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(permute_102, [512, 256, 14, 14]);  permute_102 = None
        empty_320: "bf16[512, 256, 14, 14]" = torch.ops.aten.empty.memory_format([512, 256, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_104: "bf16[512, 256, 14, 14]" = torch.ops.aten.permute.default(empty_320, [0, 1, 2, 3]);  empty_320 = None
        
        # No stacktrace found for following nodes
        as_strided_default_108: "i8[25690112]" = torch.ops.aten.as_strided.default(full_default_150, [25690112], [1], 0)
        clone_default_54: "i8[25690112]" = torch.ops.aten.clone.default(as_strided_default_108);  as_strided_default_108 = None
        as_strided_default_109: "i8[512, 256, 14, 14]" = torch.ops.aten.as_strided.default(clone_default_54, [512, 256, 14, 14], [50176, 196, 14, 1], 0);  clone_default_54 = None
        triton_kernel_wrapper_mutation_102 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 519, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_877, 'Y_ptr': permute_104, 'Mask_prt': as_strided_default_109, 'n_elts': 25690112, 'BLOCK_SIZE': 1024});  view_877 = triton_kernel_wrapper_mutation_102 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_880: "i8[512, 98, 512]" = torch.ops.aten.reshape.default(as_strided_default_109, [512, -1, 512]);  as_strided_default_109 = None
        view_881: "i8[50176, 512]" = torch.ops.aten.reshape.default(view_880, [50176, 512]);  view_880 = None
        
        # No stacktrace found for following nodes
        as_strided_default_106: "i32[802816]" = torch.ops.aten.as_strided.default(full_default_151, [802816], [1], 0)
        clone_default_53: "i32[802816]" = torch.ops.aten.clone.default(as_strided_default_106);  as_strided_default_106 = None
        as_strided_default_107: "i32[50176, 16]" = torch.ops.aten.as_strided.default(clone_default_53, [50176, 16], [16, 1], 0);  clone_default_53 = None
        triton_kernel_wrapper_mutation_101 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 520, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_881, 'P_ptr': as_strided_default_107, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_881 = triton_kernel_wrapper_mutation_101 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_37: "bf16[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_218, torch.bfloat16);  primals_218 = None
        empty_321: "i32[50176, 32]" = torch.ops.aten.empty.memory_format([50176, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_322: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_323: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_884: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(permute_104, [512, -1, 512])
        view_885: "bf16[50176, 512]" = torch.ops.aten.reshape.default(view_884, [50176, 512]);  view_884 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_100 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 362, constant_args_idx = 521, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_885, 'P_ptr': empty_321, 'S_ptr': empty_322, 'M_ptr': empty_323, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_885 = triton_kernel_wrapper_mutation_100 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_36: "bf16[512, 1024, 14, 14]" = torch.ops.aten.convolution.default(permute_104, convert_element_type_37, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_104 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_154: "i64[]" = torch.ops.aten.add.Tensor(primals_219, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_891: "bf16[512, 1024, 196]" = torch.ops.aten.reshape.default(convolution_36, [512, 1024, 196]);  convolution_36 = None
        
        # No stacktrace found for following nodes
        as_strided_default_102: "f32[1024]" = torch.ops.aten.as_strided.default(full_default_152, [1024], [1], 0)
        clone_default_51: "f32[1024]" = torch.ops.aten.clone.default(as_strided_default_102);  as_strided_default_102 = None
        as_strided_default_103: "f32[1024]" = torch.ops.aten.as_strided.default(clone_default_51, [1024], [1], 0);  clone_default_51 = None
        as_strided_default_104: "f32[1024]" = torch.ops.aten.as_strided.default(full_default_152, [1024], [1], 0)
        clone_default_52: "f32[1024]" = torch.ops.aten.clone.default(as_strided_default_104);  as_strided_default_104 = None
        as_strided_default_105: "f32[1024]" = torch.ops.aten.as_strided.default(clone_default_52, [1024], [1], 0);  clone_default_52 = None
        triton_kernel_wrapper_mutation_99 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 363, constant_args_idx = 522, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_891, 'SUM': as_strided_default_103, 'SUMSQ': as_strided_default_105, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_99 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_108: "f32[1024]" = torch.ops.aten.div.Tensor(as_strided_default_103, full_default_148);  as_strided_default_103 = None
        div_109: "f32[1024]" = torch.ops.aten.div.Tensor(as_strided_default_105, full_default_148);  as_strided_default_105 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_98 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 364, constant_args_idx = 523, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_891, 'MEAN': div_108, 'INVSTD': rsqrt_36, 'GAMMA': primals_220, 'BETA': primals_221, 'Y': permute_105, 'X_hat': permute_106, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024});  view_891 = div_108 = primals_221 = triton_kernel_wrapper_mutation_98 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_326: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_327: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_328: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_895: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(permute_106, [512, -1, 512]);  permute_106 = None
        view_896: "bf16[200704, 512]" = torch.ops.aten.reshape.default(view_895, [200704, 512]);  view_895 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_97 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 365, constant_args_idx = 524, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_896, 'P_ptr': empty_326, 'S_ptr': empty_327, 'M_ptr': empty_328, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_896 = triton_kernel_wrapper_mutation_97 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:123 in forward, code: add_10 = layer3_3_bn3 + layer3_2_relu_2;  layer3_3_bn3 = layer3_2_relu_2 = None
        view_902: "bf16[512, 1024, 14, 14]" = torch.ops.aten.reshape.default(permute_105, [512, 1024, 14, 14]);  permute_105 = None
        add_158: "bf16[512, 1024, 14, 14]" = torch.ops.aten.add.Tensor(view_902, permute_98);  view_902 = permute_98 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        empty_329: "bf16[512, 1024, 14, 14]" = torch.ops.aten.empty.memory_format([512, 1024, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_107: "bf16[512, 1024, 14, 14]" = torch.ops.aten.permute.default(empty_329, [0, 1, 2, 3]);  empty_329 = None
        
        # No stacktrace found for following nodes
        as_strided_default_100: "i8[102760448]" = torch.ops.aten.as_strided.default(full_default_160, [102760448], [1], 0)
        clone_default_50: "i8[102760448]" = torch.ops.aten.clone.default(as_strided_default_100);  as_strided_default_100 = None
        as_strided_default_101: "i8[512, 1024, 14, 14]" = torch.ops.aten.as_strided.default(clone_default_50, [512, 1024, 14, 14], [200704, 196, 14, 1], 0);  clone_default_50 = None
        triton_kernel_wrapper_mutation_96 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 525, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_158, 'Y_ptr': permute_107, 'Mask_prt': as_strided_default_101, 'n_elts': 102760448, 'BLOCK_SIZE': 1024});  add_158 = triton_kernel_wrapper_mutation_96 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_905: "i8[512, 392, 512]" = torch.ops.aten.reshape.default(as_strided_default_101, [512, -1, 512]);  as_strided_default_101 = None
        view_906: "i8[200704, 512]" = torch.ops.aten.reshape.default(view_905, [200704, 512]);  view_905 = None
        
        # No stacktrace found for following nodes
        as_strided_default_98: "i32[3211264]" = torch.ops.aten.as_strided.default(full_default_11, [3211264], [1], 0)
        clone_default_49: "i32[3211264]" = torch.ops.aten.clone.default(as_strided_default_98);  as_strided_default_98 = None
        as_strided_default_99: "i32[200704, 16]" = torch.ops.aten.as_strided.default(clone_default_49, [200704, 16], [16, 1], 0);  clone_default_49 = None
        triton_kernel_wrapper_mutation_95 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 526, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_906, 'P_ptr': as_strided_default_99, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_906 = triton_kernel_wrapper_mutation_95 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_38: "bf16[256, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_224, torch.bfloat16);  primals_224 = None
        empty_330: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_331: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_332: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_909: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(permute_107, [512, -1, 512])
        view_910: "bf16[200704, 512]" = torch.ops.aten.reshape.default(view_909, [200704, 512]);  view_909 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_94 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 366, constant_args_idx = 527, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_910, 'P_ptr': empty_330, 'S_ptr': empty_331, 'M_ptr': empty_332, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_910 = triton_kernel_wrapper_mutation_94 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_37: "bf16[512, 256, 14, 14]" = torch.ops.aten.convolution.default(permute_107, convert_element_type_38, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_159: "i64[]" = torch.ops.aten.add.Tensor(primals_225, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_916: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(convolution_37, [512, 256, 196]);  convolution_37 = None
        
        # No stacktrace found for following nodes
        as_strided_default_94: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_47: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_94);  as_strided_default_94 = None
        as_strided_default_95: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_47, [256], [1], 0);  clone_default_47 = None
        as_strided_default_96: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_48: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_96);  as_strided_default_96 = None
        as_strided_default_97: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_48, [256], [1], 0);  clone_default_48 = None
        triton_kernel_wrapper_mutation_93 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 367, constant_args_idx = 528, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_916, 'SUM': as_strided_default_95, 'SUMSQ': as_strided_default_97, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_93 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_111: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_95, full_default_148);  as_strided_default_95 = None
        div_112: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_97, full_default_148);  as_strided_default_97 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_92 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 368, constant_args_idx = 529, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_916, 'MEAN': div_111, 'INVSTD': rsqrt_37, 'GAMMA': primals_226, 'BETA': primals_227, 'Y': permute_108, 'X_hat': permute_109, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  view_916 = div_111 = primals_227 = triton_kernel_wrapper_mutation_92 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_335: "i32[50176, 32]" = torch.ops.aten.empty.memory_format([50176, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_336: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_337: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_920: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(permute_109, [512, -1, 512]);  permute_109 = None
        view_921: "bf16[50176, 512]" = torch.ops.aten.reshape.default(view_920, [50176, 512]);  view_920 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_91 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 369, constant_args_idx = 530, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_921, 'P_ptr': empty_335, 'S_ptr': empty_336, 'M_ptr': empty_337, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_921 = triton_kernel_wrapper_mutation_91 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_927: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(permute_108, [512, 256, 14, 14]);  permute_108 = None
        empty_338: "bf16[512, 256, 14, 14]" = torch.ops.aten.empty.memory_format([512, 256, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_110: "bf16[512, 256, 14, 14]" = torch.ops.aten.permute.default(empty_338, [0, 1, 2, 3]);  empty_338 = None
        
        # No stacktrace found for following nodes
        as_strided_default_92: "i8[25690112]" = torch.ops.aten.as_strided.default(full_default_150, [25690112], [1], 0)
        clone_default_46: "i8[25690112]" = torch.ops.aten.clone.default(as_strided_default_92);  as_strided_default_92 = None
        as_strided_default_93: "i8[512, 256, 14, 14]" = torch.ops.aten.as_strided.default(clone_default_46, [512, 256, 14, 14], [50176, 196, 14, 1], 0);  clone_default_46 = None
        triton_kernel_wrapper_mutation_90 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 531, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_927, 'Y_ptr': permute_110, 'Mask_prt': as_strided_default_93, 'n_elts': 25690112, 'BLOCK_SIZE': 1024});  view_927 = triton_kernel_wrapper_mutation_90 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_930: "i8[512, 98, 512]" = torch.ops.aten.reshape.default(as_strided_default_93, [512, -1, 512]);  as_strided_default_93 = None
        view_931: "i8[50176, 512]" = torch.ops.aten.reshape.default(view_930, [50176, 512]);  view_930 = None
        
        # No stacktrace found for following nodes
        as_strided_default_90: "i32[802816]" = torch.ops.aten.as_strided.default(full_default_151, [802816], [1], 0)
        clone_default_45: "i32[802816]" = torch.ops.aten.clone.default(as_strided_default_90);  as_strided_default_90 = None
        as_strided_default_91: "i32[50176, 16]" = torch.ops.aten.as_strided.default(clone_default_45, [50176, 16], [16, 1], 0);  clone_default_45 = None
        triton_kernel_wrapper_mutation_89 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 532, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_931, 'P_ptr': as_strided_default_91, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_931 = triton_kernel_wrapper_mutation_89 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_39: "bf16[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_230, torch.bfloat16);  primals_230 = None
        empty_339: "i32[50176, 32]" = torch.ops.aten.empty.memory_format([50176, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_340: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_341: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_934: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(permute_110, [512, -1, 512])
        view_935: "bf16[50176, 512]" = torch.ops.aten.reshape.default(view_934, [50176, 512]);  view_934 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_88 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 370, constant_args_idx = 533, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_935, 'P_ptr': empty_339, 'S_ptr': empty_340, 'M_ptr': empty_341, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_935 = triton_kernel_wrapper_mutation_88 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_38: "bf16[512, 256, 14, 14]" = torch.ops.aten.convolution.default(permute_110, convert_element_type_39, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  permute_110 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_163: "i64[]" = torch.ops.aten.add.Tensor(primals_231, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_941: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(convolution_38, [512, 256, 196]);  convolution_38 = None
        
        # No stacktrace found for following nodes
        as_strided_default_86: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_43: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_86);  as_strided_default_86 = None
        as_strided_default_87: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_43, [256], [1], 0);  clone_default_43 = None
        as_strided_default_88: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_44: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_88);  as_strided_default_88 = None
        as_strided_default_89: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_44, [256], [1], 0);  clone_default_44 = None
        triton_kernel_wrapper_mutation_87 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 371, constant_args_idx = 534, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_941, 'SUM': as_strided_default_87, 'SUMSQ': as_strided_default_89, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_87 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_114: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_87, full_default_148);  as_strided_default_87 = None
        div_115: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_89, full_default_148);  as_strided_default_89 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_86 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 372, constant_args_idx = 535, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_941, 'MEAN': div_114, 'INVSTD': rsqrt_38, 'GAMMA': primals_232, 'BETA': primals_233, 'Y': permute_111, 'X_hat': permute_112, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  view_941 = div_114 = primals_233 = triton_kernel_wrapper_mutation_86 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_344: "i32[50176, 32]" = torch.ops.aten.empty.memory_format([50176, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_345: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_346: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_945: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(permute_112, [512, -1, 512]);  permute_112 = None
        view_946: "bf16[50176, 512]" = torch.ops.aten.reshape.default(view_945, [50176, 512]);  view_945 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_85 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 373, constant_args_idx = 536, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_946, 'P_ptr': empty_344, 'S_ptr': empty_345, 'M_ptr': empty_346, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_946 = triton_kernel_wrapper_mutation_85 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_952: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(permute_111, [512, 256, 14, 14]);  permute_111 = None
        empty_347: "bf16[512, 256, 14, 14]" = torch.ops.aten.empty.memory_format([512, 256, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_113: "bf16[512, 256, 14, 14]" = torch.ops.aten.permute.default(empty_347, [0, 1, 2, 3]);  empty_347 = None
        
        # No stacktrace found for following nodes
        as_strided_default_84: "i8[25690112]" = torch.ops.aten.as_strided.default(full_default_150, [25690112], [1], 0)
        clone_default_42: "i8[25690112]" = torch.ops.aten.clone.default(as_strided_default_84);  as_strided_default_84 = None
        as_strided_default_85: "i8[512, 256, 14, 14]" = torch.ops.aten.as_strided.default(clone_default_42, [512, 256, 14, 14], [50176, 196, 14, 1], 0);  clone_default_42 = None
        triton_kernel_wrapper_mutation_84 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 537, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_952, 'Y_ptr': permute_113, 'Mask_prt': as_strided_default_85, 'n_elts': 25690112, 'BLOCK_SIZE': 1024});  view_952 = triton_kernel_wrapper_mutation_84 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_955: "i8[512, 98, 512]" = torch.ops.aten.reshape.default(as_strided_default_85, [512, -1, 512]);  as_strided_default_85 = None
        view_956: "i8[50176, 512]" = torch.ops.aten.reshape.default(view_955, [50176, 512]);  view_955 = None
        
        # No stacktrace found for following nodes
        as_strided_default_82: "i32[802816]" = torch.ops.aten.as_strided.default(full_default_151, [802816], [1], 0)
        clone_default_41: "i32[802816]" = torch.ops.aten.clone.default(as_strided_default_82);  as_strided_default_82 = None
        as_strided_default_83: "i32[50176, 16]" = torch.ops.aten.as_strided.default(clone_default_41, [50176, 16], [16, 1], 0);  clone_default_41 = None
        triton_kernel_wrapper_mutation_83 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 538, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_956, 'P_ptr': as_strided_default_83, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_956 = triton_kernel_wrapper_mutation_83 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_40: "bf16[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_236, torch.bfloat16);  primals_236 = None
        empty_348: "i32[50176, 32]" = torch.ops.aten.empty.memory_format([50176, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_349: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_350: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_959: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(permute_113, [512, -1, 512])
        view_960: "bf16[50176, 512]" = torch.ops.aten.reshape.default(view_959, [50176, 512]);  view_959 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_82 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 374, constant_args_idx = 539, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_960, 'P_ptr': empty_348, 'S_ptr': empty_349, 'M_ptr': empty_350, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_960 = triton_kernel_wrapper_mutation_82 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_39: "bf16[512, 1024, 14, 14]" = torch.ops.aten.convolution.default(permute_113, convert_element_type_40, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_113 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_167: "i64[]" = torch.ops.aten.add.Tensor(primals_237, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_966: "bf16[512, 1024, 196]" = torch.ops.aten.reshape.default(convolution_39, [512, 1024, 196]);  convolution_39 = None
        
        # No stacktrace found for following nodes
        as_strided_default_78: "f32[1024]" = torch.ops.aten.as_strided.default(full_default_152, [1024], [1], 0)
        clone_default_39: "f32[1024]" = torch.ops.aten.clone.default(as_strided_default_78);  as_strided_default_78 = None
        as_strided_default_79: "f32[1024]" = torch.ops.aten.as_strided.default(clone_default_39, [1024], [1], 0);  clone_default_39 = None
        as_strided_default_80: "f32[1024]" = torch.ops.aten.as_strided.default(full_default_152, [1024], [1], 0)
        clone_default_40: "f32[1024]" = torch.ops.aten.clone.default(as_strided_default_80);  as_strided_default_80 = None
        as_strided_default_81: "f32[1024]" = torch.ops.aten.as_strided.default(clone_default_40, [1024], [1], 0);  clone_default_40 = None
        triton_kernel_wrapper_mutation_81 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 375, constant_args_idx = 540, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_966, 'SUM': as_strided_default_79, 'SUMSQ': as_strided_default_81, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_81 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_117: "f32[1024]" = torch.ops.aten.div.Tensor(as_strided_default_79, full_default_148);  as_strided_default_79 = None
        div_118: "f32[1024]" = torch.ops.aten.div.Tensor(as_strided_default_81, full_default_148);  as_strided_default_81 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_80 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 376, constant_args_idx = 541, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_966, 'MEAN': div_117, 'INVSTD': rsqrt_39, 'GAMMA': primals_238, 'BETA': primals_239, 'Y': permute_114, 'X_hat': permute_115, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024});  view_966 = div_117 = primals_239 = triton_kernel_wrapper_mutation_80 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_353: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_354: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_355: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_970: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(permute_115, [512, -1, 512]);  permute_115 = None
        view_971: "bf16[200704, 512]" = torch.ops.aten.reshape.default(view_970, [200704, 512]);  view_970 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_79 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 377, constant_args_idx = 542, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_971, 'P_ptr': empty_353, 'S_ptr': empty_354, 'M_ptr': empty_355, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_971 = triton_kernel_wrapper_mutation_79 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:133 in forward, code: add_11 = layer3_4_bn3 + layer3_3_relu_2;  layer3_4_bn3 = layer3_3_relu_2 = None
        view_977: "bf16[512, 1024, 14, 14]" = torch.ops.aten.reshape.default(permute_114, [512, 1024, 14, 14]);  permute_114 = None
        add_171: "bf16[512, 1024, 14, 14]" = torch.ops.aten.add.Tensor(view_977, permute_107);  view_977 = permute_107 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        empty_356: "bf16[512, 1024, 14, 14]" = torch.ops.aten.empty.memory_format([512, 1024, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_116: "bf16[512, 1024, 14, 14]" = torch.ops.aten.permute.default(empty_356, [0, 1, 2, 3]);  empty_356 = None
        
        # No stacktrace found for following nodes
        as_strided_default_76: "i8[102760448]" = torch.ops.aten.as_strided.default(full_default_160, [102760448], [1], 0)
        clone_default_38: "i8[102760448]" = torch.ops.aten.clone.default(as_strided_default_76);  as_strided_default_76 = None
        as_strided_default_77: "i8[512, 1024, 14, 14]" = torch.ops.aten.as_strided.default(clone_default_38, [512, 1024, 14, 14], [200704, 196, 14, 1], 0);  clone_default_38 = None
        triton_kernel_wrapper_mutation_78 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 543, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_171, 'Y_ptr': permute_116, 'Mask_prt': as_strided_default_77, 'n_elts': 102760448, 'BLOCK_SIZE': 1024});  add_171 = triton_kernel_wrapper_mutation_78 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_980: "i8[512, 392, 512]" = torch.ops.aten.reshape.default(as_strided_default_77, [512, -1, 512]);  as_strided_default_77 = None
        view_981: "i8[200704, 512]" = torch.ops.aten.reshape.default(view_980, [200704, 512]);  view_980 = None
        
        # No stacktrace found for following nodes
        as_strided_default_74: "i32[3211264]" = torch.ops.aten.as_strided.default(full_default_11, [3211264], [1], 0)
        clone_default_37: "i32[3211264]" = torch.ops.aten.clone.default(as_strided_default_74);  as_strided_default_74 = None
        as_strided_default_75: "i32[200704, 16]" = torch.ops.aten.as_strided.default(clone_default_37, [200704, 16], [16, 1], 0);  clone_default_37 = None
        triton_kernel_wrapper_mutation_77 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 544, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_981, 'P_ptr': as_strided_default_75, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_981 = triton_kernel_wrapper_mutation_77 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_41: "bf16[256, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_242, torch.bfloat16);  primals_242 = None
        empty_357: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_358: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_359: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_984: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(permute_116, [512, -1, 512])
        view_985: "bf16[200704, 512]" = torch.ops.aten.reshape.default(view_984, [200704, 512]);  view_984 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_76 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 378, constant_args_idx = 545, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_985, 'P_ptr': empty_357, 'S_ptr': empty_358, 'M_ptr': empty_359, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_985 = triton_kernel_wrapper_mutation_76 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_40: "bf16[512, 256, 14, 14]" = torch.ops.aten.convolution.default(permute_116, convert_element_type_41, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_172: "i64[]" = torch.ops.aten.add.Tensor(primals_243, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_991: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(convolution_40, [512, 256, 196]);  convolution_40 = None
        
        # No stacktrace found for following nodes
        as_strided_default_70: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_35: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_70);  as_strided_default_70 = None
        as_strided_default_71: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_35, [256], [1], 0);  clone_default_35 = None
        as_strided_default_72: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_36: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_72);  as_strided_default_72 = None
        as_strided_default_73: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_36, [256], [1], 0);  clone_default_36 = None
        triton_kernel_wrapper_mutation_75 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 379, constant_args_idx = 546, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_991, 'SUM': as_strided_default_71, 'SUMSQ': as_strided_default_73, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_75 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_120: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_71, full_default_148);  as_strided_default_71 = None
        div_121: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_73, full_default_148);  as_strided_default_73 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_74 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 380, constant_args_idx = 547, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_991, 'MEAN': div_120, 'INVSTD': rsqrt_40, 'GAMMA': primals_244, 'BETA': primals_245, 'Y': permute_117, 'X_hat': permute_118, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  view_991 = div_120 = primals_245 = triton_kernel_wrapper_mutation_74 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_362: "i32[50176, 32]" = torch.ops.aten.empty.memory_format([50176, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_363: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_364: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_995: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(permute_118, [512, -1, 512]);  permute_118 = None
        view_996: "bf16[50176, 512]" = torch.ops.aten.reshape.default(view_995, [50176, 512]);  view_995 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_73 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 381, constant_args_idx = 548, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_996, 'P_ptr': empty_362, 'S_ptr': empty_363, 'M_ptr': empty_364, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_996 = triton_kernel_wrapper_mutation_73 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1002: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(permute_117, [512, 256, 14, 14]);  permute_117 = None
        empty_365: "bf16[512, 256, 14, 14]" = torch.ops.aten.empty.memory_format([512, 256, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_119: "bf16[512, 256, 14, 14]" = torch.ops.aten.permute.default(empty_365, [0, 1, 2, 3]);  empty_365 = None
        
        # No stacktrace found for following nodes
        as_strided_default_68: "i8[25690112]" = torch.ops.aten.as_strided.default(full_default_150, [25690112], [1], 0)
        clone_default_34: "i8[25690112]" = torch.ops.aten.clone.default(as_strided_default_68);  as_strided_default_68 = None
        as_strided_default_69: "i8[512, 256, 14, 14]" = torch.ops.aten.as_strided.default(clone_default_34, [512, 256, 14, 14], [50176, 196, 14, 1], 0);  clone_default_34 = None
        triton_kernel_wrapper_mutation_72 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 549, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1002, 'Y_ptr': permute_119, 'Mask_prt': as_strided_default_69, 'n_elts': 25690112, 'BLOCK_SIZE': 1024});  view_1002 = triton_kernel_wrapper_mutation_72 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1005: "i8[512, 98, 512]" = torch.ops.aten.reshape.default(as_strided_default_69, [512, -1, 512]);  as_strided_default_69 = None
        view_1006: "i8[50176, 512]" = torch.ops.aten.reshape.default(view_1005, [50176, 512]);  view_1005 = None
        
        # No stacktrace found for following nodes
        as_strided_default_66: "i32[802816]" = torch.ops.aten.as_strided.default(full_default_151, [802816], [1], 0)
        clone_default_33: "i32[802816]" = torch.ops.aten.clone.default(as_strided_default_66);  as_strided_default_66 = None
        as_strided_default_67: "i32[50176, 16]" = torch.ops.aten.as_strided.default(clone_default_33, [50176, 16], [16, 1], 0);  clone_default_33 = None
        triton_kernel_wrapper_mutation_71 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 550, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1006, 'P_ptr': as_strided_default_67, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_1006 = triton_kernel_wrapper_mutation_71 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_42: "bf16[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_248, torch.bfloat16);  primals_248 = None
        empty_366: "i32[50176, 32]" = torch.ops.aten.empty.memory_format([50176, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_367: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_368: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1009: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(permute_119, [512, -1, 512])
        view_1010: "bf16[50176, 512]" = torch.ops.aten.reshape.default(view_1009, [50176, 512]);  view_1009 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_70 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 382, constant_args_idx = 551, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1010, 'P_ptr': empty_366, 'S_ptr': empty_367, 'M_ptr': empty_368, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_1010 = triton_kernel_wrapper_mutation_70 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_41: "bf16[512, 256, 14, 14]" = torch.ops.aten.convolution.default(permute_119, convert_element_type_42, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  permute_119 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_176: "i64[]" = torch.ops.aten.add.Tensor(primals_249, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_1016: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(convolution_41, [512, 256, 196]);  convolution_41 = None
        
        # No stacktrace found for following nodes
        as_strided_default_64: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_32: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_64);  as_strided_default_64 = None
        as_strided_default_65: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_32, [256], [1], 0);  clone_default_32 = None
        triton_kernel_wrapper_mutation_69 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 383, constant_args_idx = 552, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1016, 'SUM': full_default_18, 'SUMSQ': as_strided_default_65, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_69 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_123: "f32[256]" = torch.ops.aten.div.Tensor(full_default_18, full_default_148);  full_default_18 = None
        div_124: "f32[256]" = torch.ops.aten.div.Tensor(as_strided_default_65, full_default_148);  as_strided_default_65 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_68 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 384, constant_args_idx = 553, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1016, 'MEAN': div_123, 'INVSTD': rsqrt_41, 'GAMMA': primals_250, 'BETA': primals_251, 'Y': permute_120, 'X_hat': permute_121, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  view_1016 = div_123 = primals_251 = triton_kernel_wrapper_mutation_68 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_371: "i32[50176, 32]" = torch.ops.aten.empty.memory_format([50176, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_372: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_373: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1020: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(permute_121, [512, -1, 512]);  permute_121 = None
        view_1021: "bf16[50176, 512]" = torch.ops.aten.reshape.default(view_1020, [50176, 512]);  view_1020 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_67 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 385, constant_args_idx = 554, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1021, 'P_ptr': empty_371, 'S_ptr': empty_372, 'M_ptr': empty_373, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_1021 = triton_kernel_wrapper_mutation_67 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1027: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(permute_120, [512, 256, 14, 14]);  permute_120 = None
        empty_374: "bf16[512, 256, 14, 14]" = torch.ops.aten.empty.memory_format([512, 256, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_122: "bf16[512, 256, 14, 14]" = torch.ops.aten.permute.default(empty_374, [0, 1, 2, 3]);  empty_374 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_66 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 555, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1027, 'Y_ptr': permute_122, 'Mask_prt': full_default_150, 'n_elts': 25690112, 'BLOCK_SIZE': 1024});  view_1027 = triton_kernel_wrapper_mutation_66 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1030: "i8[512, 98, 512]" = torch.ops.aten.reshape.default(full_default_150, [512, -1, 512]);  full_default_150 = None
        view_1031: "i8[50176, 512]" = torch.ops.aten.reshape.default(view_1030, [50176, 512]);  view_1030 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_65 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 556, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1031, 'P_ptr': full_default_151, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_1031 = triton_kernel_wrapper_mutation_65 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_43: "bf16[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_254, torch.bfloat16);  primals_254 = None
        empty_375: "i32[50176, 32]" = torch.ops.aten.empty.memory_format([50176, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_376: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_377: "bf16[50176]" = torch.ops.aten.empty.memory_format([50176], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1034: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(permute_122, [512, -1, 512])
        view_1035: "bf16[50176, 512]" = torch.ops.aten.reshape.default(view_1034, [50176, 512]);  view_1034 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_64 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 386, constant_args_idx = 557, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1035, 'P_ptr': empty_375, 'S_ptr': empty_376, 'M_ptr': empty_377, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_1035 = triton_kernel_wrapper_mutation_64 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_42: "bf16[512, 1024, 14, 14]" = torch.ops.aten.convolution.default(permute_122, convert_element_type_43, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_122 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_180: "i64[]" = torch.ops.aten.add.Tensor(primals_255, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_1041: "bf16[512, 1024, 196]" = torch.ops.aten.reshape.default(convolution_42, [512, 1024, 196]);  convolution_42 = None
        
        # No stacktrace found for following nodes
        as_strided_default_62: "f32[1024]" = torch.ops.aten.as_strided.default(full_default_152, [1024], [1], 0)
        clone_default_31: "f32[1024]" = torch.ops.aten.clone.default(as_strided_default_62);  as_strided_default_62 = None
        as_strided_default_63: "f32[1024]" = torch.ops.aten.as_strided.default(clone_default_31, [1024], [1], 0);  clone_default_31 = None
        triton_kernel_wrapper_mutation_63 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 387, constant_args_idx = 558, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1041, 'SUM': full_default_152, 'SUMSQ': as_strided_default_63, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_63 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_126: "f32[1024]" = torch.ops.aten.div.Tensor(full_default_152, full_default_148);  full_default_152 = None
        div_127: "f32[1024]" = torch.ops.aten.div.Tensor(as_strided_default_63, full_default_148);  as_strided_default_63 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_62 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 388, constant_args_idx = 559, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1041, 'MEAN': div_126, 'INVSTD': rsqrt_42, 'GAMMA': primals_256, 'BETA': primals_257, 'Y': permute_123, 'X_hat': permute_124, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024});  view_1041 = div_126 = primals_257 = triton_kernel_wrapper_mutation_62 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_380: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_381: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_382: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1045: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(permute_124, [512, -1, 512]);  permute_124 = None
        view_1046: "bf16[200704, 512]" = torch.ops.aten.reshape.default(view_1045, [200704, 512]);  view_1045 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_61 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 389, constant_args_idx = 560, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1046, 'P_ptr': empty_380, 'S_ptr': empty_381, 'M_ptr': empty_382, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_1046 = triton_kernel_wrapper_mutation_61 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:143 in forward, code: add_12 = layer3_5_bn3 + layer3_4_relu_2;  layer3_5_bn3 = layer3_4_relu_2 = None
        view_1052: "bf16[512, 1024, 14, 14]" = torch.ops.aten.reshape.default(permute_123, [512, 1024, 14, 14]);  permute_123 = None
        add_184: "bf16[512, 1024, 14, 14]" = torch.ops.aten.add.Tensor(view_1052, permute_116);  view_1052 = permute_116 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        empty_383: "bf16[512, 1024, 14, 14]" = torch.ops.aten.empty.memory_format([512, 1024, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_125: "bf16[512, 1024, 14, 14]" = torch.ops.aten.permute.default(empty_383, [0, 1, 2, 3]);  empty_383 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_60 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 561, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_184, 'Y_ptr': permute_125, 'Mask_prt': full_default_160, 'n_elts': 102760448, 'BLOCK_SIZE': 1024});  add_184 = triton_kernel_wrapper_mutation_60 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1055: "i8[512, 392, 512]" = torch.ops.aten.reshape.default(full_default_160, [512, -1, 512]);  full_default_160 = None
        view_1056: "i8[200704, 512]" = torch.ops.aten.reshape.default(view_1055, [200704, 512]);  view_1055 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_59 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 562, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1056, 'P_ptr': full_default_11, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_1056 = triton_kernel_wrapper_mutation_59 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_44: "bf16[512, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_260, torch.bfloat16);  primals_260 = None
        empty_384: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_385: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_386: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1059: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(permute_125, [512, -1, 512])
        view_1060: "bf16[200704, 512]" = torch.ops.aten.reshape.default(view_1059, [200704, 512]);  view_1059 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_58 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 390, constant_args_idx = 563, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1060, 'P_ptr': empty_384, 'S_ptr': empty_385, 'M_ptr': empty_386, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  triton_kernel_wrapper_mutation_58 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_43: "bf16[512, 512, 14, 14]" = torch.ops.aten.convolution.default(permute_125, convert_element_type_44, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_185: "i64[]" = torch.ops.aten.add.Tensor(primals_261, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_1066: "bf16[512, 512, 196]" = torch.ops.aten.reshape.default(convolution_43, [512, 512, 196]);  convolution_43 = None
        
        # No stacktrace found for following nodes
        as_strided_default_58: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_29: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_58);  as_strided_default_58 = None
        as_strided_default_59: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_29, [512], [1], 0);  clone_default_29 = None
        as_strided_default_60: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_30: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_60);  as_strided_default_60 = None
        as_strided_default_61: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_30, [512], [1], 0);  clone_default_30 = None
        triton_kernel_wrapper_mutation_57 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 391, constant_args_idx = 564, grid = [(512, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1066, 'SUM': as_strided_default_59, 'SUMSQ': as_strided_default_61, 'M': 100352, 'HW': 196, 'stride_n': 100352, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_57 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_129: "f32[512]" = torch.ops.aten.div.Tensor(as_strided_default_59, full_default_148);  as_strided_default_59 = None
        div_130: "f32[512]" = torch.ops.aten.div.Tensor(as_strided_default_61, full_default_148);  as_strided_default_61 = full_default_148 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_56 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 392, constant_args_idx = 565, grid = [(512, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1066, 'MEAN': div_129, 'INVSTD': rsqrt_43, 'GAMMA': primals_262, 'BETA': primals_263, 'Y': permute_126, 'X_hat': permute_127, 'M': 100352, 'HW': 196, 'stride_n': 100352, 'stride_c': 196, 'BLOCK_M': 1024});  view_1066 = div_129 = primals_263 = triton_kernel_wrapper_mutation_56 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_389: "i32[100352, 32]" = torch.ops.aten.empty.memory_format([100352, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_390: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_391: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1070: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(permute_127, [512, -1, 512]);  permute_127 = None
        view_1071: "bf16[100352, 512]" = torch.ops.aten.reshape.default(view_1070, [100352, 512]);  view_1070 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_55 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 393, constant_args_idx = 566, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1071, 'P_ptr': empty_389, 'S_ptr': empty_390, 'M_ptr': empty_391, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_1071 = triton_kernel_wrapper_mutation_55 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_256: "i8[512, 512, 14, 14]" = torch.ops.aten.full.default([512, 512, 14, 14], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_1077: "bf16[512, 512, 14, 14]" = torch.ops.aten.reshape.default(permute_126, [512, 512, 14, 14]);  permute_126 = None
        empty_392: "bf16[512, 512, 14, 14]" = torch.ops.aten.empty.memory_format([512, 512, 14, 14], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_128: "bf16[512, 512, 14, 14]" = torch.ops.aten.permute.default(empty_392, [0, 1, 2, 3]);  empty_392 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_54 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 567, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1077, 'Y_ptr': permute_128, 'Mask_prt': full_default_256, 'n_elts': 51380224, 'BLOCK_SIZE': 1024});  view_1077 = triton_kernel_wrapper_mutation_54 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1080: "i8[512, 196, 512]" = torch.ops.aten.reshape.default(full_default_256, [512, -1, 512]);  full_default_256 = None
        view_1081: "i8[100352, 512]" = torch.ops.aten.reshape.default(view_1080, [100352, 512]);  view_1080 = None
        
        # No stacktrace found for following nodes
        as_strided_default_56: "i32[1605632]" = torch.ops.aten.as_strided.default(full_default_75, [1605632], [1], 0)
        clone_default_28: "i32[1605632]" = torch.ops.aten.clone.default(as_strided_default_56);  as_strided_default_56 = None
        as_strided_default_57: "i32[100352, 16]" = torch.ops.aten.as_strided.default(clone_default_28, [100352, 16], [16, 1], 0);  clone_default_28 = None
        triton_kernel_wrapper_mutation_53 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 568, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1081, 'P_ptr': as_strided_default_57, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_1081 = triton_kernel_wrapper_mutation_53 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_45: "bf16[512, 512, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_266, torch.bfloat16);  primals_266 = None
        empty_393: "i32[100352, 32]" = torch.ops.aten.empty.memory_format([100352, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_394: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_395: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1084: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(permute_128, [512, -1, 512])
        view_1085: "bf16[100352, 512]" = torch.ops.aten.reshape.default(view_1084, [100352, 512]);  view_1084 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_52 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 394, constant_args_idx = 569, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1085, 'P_ptr': empty_393, 'S_ptr': empty_394, 'M_ptr': empty_395, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_1085 = triton_kernel_wrapper_mutation_52 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_44: "bf16[512, 512, 7, 7]" = torch.ops.aten.convolution.default(permute_128, convert_element_type_45, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  permute_128 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_189: "i64[]" = torch.ops.aten.add.Tensor(primals_267, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_1091: "bf16[512, 512, 49]" = torch.ops.aten.reshape.default(convolution_44, [512, 512, 49]);  convolution_44 = None
        
        # No stacktrace found for following nodes
        as_strided_default_52: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_26: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_52);  as_strided_default_52 = None
        as_strided_default_53: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_26, [512], [1], 0);  clone_default_26 = None
        as_strided_default_54: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_27: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_54);  as_strided_default_54 = None
        as_strided_default_55: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_27, [512], [1], 0);  clone_default_27 = None
        triton_kernel_wrapper_mutation_51 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 395, constant_args_idx = 570, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1091, 'SUM': as_strided_default_53, 'SUMSQ': as_strided_default_55, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_51 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        full_default_260: "f32[]" = torch.ops.aten.full.default([], 25088.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        div_132: "f32[512]" = torch.ops.aten.div.Tensor(as_strided_default_53, full_default_260);  as_strided_default_53 = None
        div_133: "f32[512]" = torch.ops.aten.div.Tensor(as_strided_default_55, full_default_260);  as_strided_default_55 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_50 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 396, constant_args_idx = 571, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1091, 'MEAN': div_132, 'INVSTD': rsqrt_44, 'GAMMA': primals_268, 'BETA': primals_269, 'Y': permute_129, 'X_hat': permute_130, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024});  view_1091 = div_132 = primals_269 = triton_kernel_wrapper_mutation_50 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_398: "i32[25088, 32]" = torch.ops.aten.empty.memory_format([25088, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_399: "bf16[25088]" = torch.ops.aten.empty.memory_format([25088], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_400: "bf16[25088]" = torch.ops.aten.empty.memory_format([25088], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1095: "bf16[512, 49, 512]" = torch.ops.aten.reshape.default(permute_130, [512, -1, 512]);  permute_130 = None
        view_1096: "bf16[25088, 512]" = torch.ops.aten.reshape.default(view_1095, [25088, 512]);  view_1095 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_49 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 397, constant_args_idx = 572, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1096, 'P_ptr': empty_398, 'S_ptr': empty_399, 'M_ptr': empty_400, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_1096 = triton_kernel_wrapper_mutation_49 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_262: "i8[512, 512, 7, 7]" = torch.ops.aten.full.default([512, 512, 7, 7], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_1102: "bf16[512, 512, 7, 7]" = torch.ops.aten.reshape.default(permute_129, [512, 512, 7, 7]);  permute_129 = None
        empty_401: "bf16[512, 512, 7, 7]" = torch.ops.aten.empty.memory_format([512, 512, 7, 7], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_131: "bf16[512, 512, 7, 7]" = torch.ops.aten.permute.default(empty_401, [0, 1, 2, 3]);  empty_401 = None
        
        # No stacktrace found for following nodes
        as_strided_default_50: "i8[12845056]" = torch.ops.aten.as_strided.default(full_default_262, [12845056], [1], 0)
        clone_default_25: "i8[12845056]" = torch.ops.aten.clone.default(as_strided_default_50);  as_strided_default_50 = None
        as_strided_default_51: "i8[512, 512, 7, 7]" = torch.ops.aten.as_strided.default(clone_default_25, [512, 512, 7, 7], [25088, 49, 7, 1], 0);  clone_default_25 = None
        triton_kernel_wrapper_mutation_48 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 573, grid = [(12544, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1102, 'Y_ptr': permute_131, 'Mask_prt': as_strided_default_51, 'n_elts': 12845056, 'BLOCK_SIZE': 1024});  view_1102 = triton_kernel_wrapper_mutation_48 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_263: "i32[25088, 16]" = torch.ops.aten.full.default([25088, 16], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_1105: "i8[512, 49, 512]" = torch.ops.aten.reshape.default(as_strided_default_51, [512, -1, 512]);  as_strided_default_51 = None
        view_1106: "i8[25088, 512]" = torch.ops.aten.reshape.default(view_1105, [25088, 512]);  view_1105 = None
        
        # No stacktrace found for following nodes
        as_strided_default_48: "i32[401408]" = torch.ops.aten.as_strided.default(full_default_263, [401408], [1], 0)
        clone_default_24: "i32[401408]" = torch.ops.aten.clone.default(as_strided_default_48);  as_strided_default_48 = None
        as_strided_default_49: "i32[25088, 16]" = torch.ops.aten.as_strided.default(clone_default_24, [25088, 16], [16, 1], 0);  clone_default_24 = None
        triton_kernel_wrapper_mutation_47 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 574, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1106, 'P_ptr': as_strided_default_49, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_1106 = triton_kernel_wrapper_mutation_47 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_46: "bf16[2048, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_272, torch.bfloat16);  primals_272 = None
        empty_402: "i32[25088, 32]" = torch.ops.aten.empty.memory_format([25088, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_403: "bf16[25088]" = torch.ops.aten.empty.memory_format([25088], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_404: "bf16[25088]" = torch.ops.aten.empty.memory_format([25088], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1109: "bf16[512, 49, 512]" = torch.ops.aten.reshape.default(permute_131, [512, -1, 512])
        view_1110: "bf16[25088, 512]" = torch.ops.aten.reshape.default(view_1109, [25088, 512]);  view_1109 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_46 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 398, constant_args_idx = 575, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1110, 'P_ptr': empty_402, 'S_ptr': empty_403, 'M_ptr': empty_404, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_1110 = triton_kernel_wrapper_mutation_46 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_45: "bf16[512, 2048, 7, 7]" = torch.ops.aten.convolution.default(permute_131, convert_element_type_46, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_131 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_193: "i64[]" = torch.ops.aten.add.Tensor(primals_273, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_1116: "bf16[512, 2048, 49]" = torch.ops.aten.reshape.default(convolution_45, [512, 2048, 49]);  convolution_45 = None
        full_default_264: "f32[2048]" = torch.ops.aten.full.default([2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        as_strided_default_44: "f32[2048]" = torch.ops.aten.as_strided.default(full_default_264, [2048], [1], 0)
        clone_default_22: "f32[2048]" = torch.ops.aten.clone.default(as_strided_default_44);  as_strided_default_44 = None
        as_strided_default_45: "f32[2048]" = torch.ops.aten.as_strided.default(clone_default_22, [2048], [1], 0);  clone_default_22 = None
        as_strided_default_46: "f32[2048]" = torch.ops.aten.as_strided.default(full_default_264, [2048], [1], 0)
        clone_default_23: "f32[2048]" = torch.ops.aten.clone.default(as_strided_default_46);  as_strided_default_46 = None
        as_strided_default_47: "f32[2048]" = torch.ops.aten.as_strided.default(clone_default_23, [2048], [1], 0);  clone_default_23 = None
        triton_kernel_wrapper_mutation_45 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 399, constant_args_idx = 576, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1116, 'SUM': as_strided_default_45, 'SUMSQ': as_strided_default_47, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_45 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_135: "f32[2048]" = torch.ops.aten.div.Tensor(as_strided_default_45, full_default_260);  as_strided_default_45 = None
        div_136: "f32[2048]" = torch.ops.aten.div.Tensor(as_strided_default_47, full_default_260);  as_strided_default_47 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_44 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 400, constant_args_idx = 577, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1116, 'MEAN': div_135, 'INVSTD': rsqrt_45, 'GAMMA': primals_274, 'BETA': primals_275, 'Y': permute_132, 'X_hat': permute_133, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024});  view_1116 = div_135 = primals_275 = triton_kernel_wrapper_mutation_44 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_407: "i32[100352, 32]" = torch.ops.aten.empty.memory_format([100352, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_408: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_409: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1120: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(permute_133, [512, -1, 512]);  permute_133 = None
        view_1121: "bf16[100352, 512]" = torch.ops.aten.reshape.default(view_1120, [100352, 512]);  view_1120 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_43 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 401, constant_args_idx = 578, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1121, 'P_ptr': empty_407, 'S_ptr': empty_408, 'M_ptr': empty_409, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_1121 = triton_kernel_wrapper_mutation_43 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_47: "bf16[2048, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_278, torch.bfloat16);  primals_278 = None
        empty_410: "i32[200704, 32]" = torch.ops.aten.empty.memory_format([200704, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_411: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_412: "bf16[200704]" = torch.ops.aten.empty.memory_format([200704], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_42 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 402, constant_args_idx = 579, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1060, 'P_ptr': empty_410, 'S_ptr': empty_411, 'M_ptr': empty_412, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_1060 = triton_kernel_wrapper_mutation_42 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_46: "bf16[512, 2048, 7, 7]" = torch.ops.aten.convolution.default(permute_125, convert_element_type_47, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  permute_125 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_197: "i64[]" = torch.ops.aten.add.Tensor(primals_279, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_1136: "bf16[512, 2048, 49]" = torch.ops.aten.reshape.default(convolution_46, [512, 2048, 49]);  convolution_46 = None
        
        # No stacktrace found for following nodes
        as_strided_default_40: "f32[2048]" = torch.ops.aten.as_strided.default(full_default_264, [2048], [1], 0)
        clone_default_20: "f32[2048]" = torch.ops.aten.clone.default(as_strided_default_40);  as_strided_default_40 = None
        as_strided_default_41: "f32[2048]" = torch.ops.aten.as_strided.default(clone_default_20, [2048], [1], 0);  clone_default_20 = None
        as_strided_default_42: "f32[2048]" = torch.ops.aten.as_strided.default(full_default_264, [2048], [1], 0)
        clone_default_21: "f32[2048]" = torch.ops.aten.clone.default(as_strided_default_42);  as_strided_default_42 = None
        as_strided_default_43: "f32[2048]" = torch.ops.aten.as_strided.default(clone_default_21, [2048], [1], 0);  clone_default_21 = None
        triton_kernel_wrapper_mutation_41 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 403, constant_args_idx = 580, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1136, 'SUM': as_strided_default_41, 'SUMSQ': as_strided_default_43, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_41 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_138: "f32[2048]" = torch.ops.aten.div.Tensor(as_strided_default_41, full_default_260);  as_strided_default_41 = None
        div_139: "f32[2048]" = torch.ops.aten.div.Tensor(as_strided_default_43, full_default_260);  as_strided_default_43 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_40 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 404, constant_args_idx = 581, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1136, 'MEAN': div_138, 'INVSTD': rsqrt_46, 'GAMMA': primals_280, 'BETA': primals_281, 'Y': permute_134, 'X_hat': permute_135, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024});  view_1136 = div_138 = primals_281 = triton_kernel_wrapper_mutation_40 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_415: "i32[100352, 32]" = torch.ops.aten.empty.memory_format([100352, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_416: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_417: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1140: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(permute_135, [512, -1, 512]);  permute_135 = None
        view_1141: "bf16[100352, 512]" = torch.ops.aten.reshape.default(view_1140, [100352, 512]);  view_1140 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_39 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 405, constant_args_idx = 582, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1141, 'P_ptr': empty_415, 'S_ptr': empty_416, 'M_ptr': empty_417, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_1141 = triton_kernel_wrapper_mutation_39 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:155 in forward, code: add_13 = layer4_0_bn3 + layer4_0_downsample_1;  layer4_0_bn3 = layer4_0_downsample_1 = None
        view_1147: "bf16[512, 2048, 7, 7]" = torch.ops.aten.reshape.default(permute_132, [512, 2048, 7, 7]);  permute_132 = None
        view_1148: "bf16[512, 2048, 7, 7]" = torch.ops.aten.reshape.default(permute_134, [512, 2048, 7, 7]);  permute_134 = None
        add_201: "bf16[512, 2048, 7, 7]" = torch.ops.aten.add.Tensor(view_1147, view_1148);  view_1147 = view_1148 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_272: "i8[512, 2048, 7, 7]" = torch.ops.aten.full.default([512, 2048, 7, 7], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        empty_418: "bf16[512, 2048, 7, 7]" = torch.ops.aten.empty.memory_format([512, 2048, 7, 7], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_136: "bf16[512, 2048, 7, 7]" = torch.ops.aten.permute.default(empty_418, [0, 1, 2, 3]);  empty_418 = None
        
        # No stacktrace found for following nodes
        as_strided_default_38: "i8[51380224]" = torch.ops.aten.as_strided.default(full_default_272, [51380224], [1], 0)
        clone_default_19: "i8[51380224]" = torch.ops.aten.clone.default(as_strided_default_38);  as_strided_default_38 = None
        as_strided_default_39: "i8[512, 2048, 7, 7]" = torch.ops.aten.as_strided.default(clone_default_19, [512, 2048, 7, 7], [100352, 49, 7, 1], 0);  clone_default_19 = None
        triton_kernel_wrapper_mutation_38 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 583, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_201, 'Y_ptr': permute_136, 'Mask_prt': as_strided_default_39, 'n_elts': 51380224, 'BLOCK_SIZE': 1024});  add_201 = triton_kernel_wrapper_mutation_38 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1151: "i8[512, 196, 512]" = torch.ops.aten.reshape.default(as_strided_default_39, [512, -1, 512]);  as_strided_default_39 = None
        view_1152: "i8[100352, 512]" = torch.ops.aten.reshape.default(view_1151, [100352, 512]);  view_1151 = None
        
        # No stacktrace found for following nodes
        as_strided_default_36: "i32[1605632]" = torch.ops.aten.as_strided.default(full_default_75, [1605632], [1], 0)
        clone_default_18: "i32[1605632]" = torch.ops.aten.clone.default(as_strided_default_36);  as_strided_default_36 = None
        as_strided_default_37: "i32[100352, 16]" = torch.ops.aten.as_strided.default(clone_default_18, [100352, 16], [16, 1], 0);  clone_default_18 = None
        triton_kernel_wrapper_mutation_37 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 584, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1152, 'P_ptr': as_strided_default_37, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_1152 = triton_kernel_wrapper_mutation_37 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_48: "bf16[512, 2048, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_284, torch.bfloat16);  primals_284 = None
        empty_419: "i32[100352, 32]" = torch.ops.aten.empty.memory_format([100352, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_420: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_421: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1155: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(permute_136, [512, -1, 512])
        view_1156: "bf16[100352, 512]" = torch.ops.aten.reshape.default(view_1155, [100352, 512]);  view_1155 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_36 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 406, constant_args_idx = 585, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1156, 'P_ptr': empty_419, 'S_ptr': empty_420, 'M_ptr': empty_421, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_1156 = triton_kernel_wrapper_mutation_36 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_47: "bf16[512, 512, 7, 7]" = torch.ops.aten.convolution.default(permute_136, convert_element_type_48, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_202: "i64[]" = torch.ops.aten.add.Tensor(primals_285, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_1162: "bf16[512, 512, 49]" = torch.ops.aten.reshape.default(convolution_47, [512, 512, 49]);  convolution_47 = None
        
        # No stacktrace found for following nodes
        as_strided_default_32: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_16: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_32);  as_strided_default_32 = None
        as_strided_default_33: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_16, [512], [1], 0);  clone_default_16 = None
        as_strided_default_34: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_17: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_34);  as_strided_default_34 = None
        as_strided_default_35: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_17, [512], [1], 0);  clone_default_17 = None
        triton_kernel_wrapper_mutation_35 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 407, constant_args_idx = 586, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1162, 'SUM': as_strided_default_33, 'SUMSQ': as_strided_default_35, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_35 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_141: "f32[512]" = torch.ops.aten.div.Tensor(as_strided_default_33, full_default_260);  as_strided_default_33 = None
        div_142: "f32[512]" = torch.ops.aten.div.Tensor(as_strided_default_35, full_default_260);  as_strided_default_35 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_34 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 408, constant_args_idx = 587, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1162, 'MEAN': div_141, 'INVSTD': rsqrt_47, 'GAMMA': primals_286, 'BETA': primals_287, 'Y': permute_137, 'X_hat': permute_138, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024});  view_1162 = div_141 = primals_287 = triton_kernel_wrapper_mutation_34 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_424: "i32[25088, 32]" = torch.ops.aten.empty.memory_format([25088, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_425: "bf16[25088]" = torch.ops.aten.empty.memory_format([25088], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_426: "bf16[25088]" = torch.ops.aten.empty.memory_format([25088], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1166: "bf16[512, 49, 512]" = torch.ops.aten.reshape.default(permute_138, [512, -1, 512]);  permute_138 = None
        view_1167: "bf16[25088, 512]" = torch.ops.aten.reshape.default(view_1166, [25088, 512]);  view_1166 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_33 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 409, constant_args_idx = 588, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1167, 'P_ptr': empty_424, 'S_ptr': empty_425, 'M_ptr': empty_426, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_1167 = triton_kernel_wrapper_mutation_33 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1173: "bf16[512, 512, 7, 7]" = torch.ops.aten.reshape.default(permute_137, [512, 512, 7, 7]);  permute_137 = None
        empty_427: "bf16[512, 512, 7, 7]" = torch.ops.aten.empty.memory_format([512, 512, 7, 7], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_139: "bf16[512, 512, 7, 7]" = torch.ops.aten.permute.default(empty_427, [0, 1, 2, 3]);  empty_427 = None
        
        # No stacktrace found for following nodes
        as_strided_default_30: "i8[12845056]" = torch.ops.aten.as_strided.default(full_default_262, [12845056], [1], 0)
        clone_default_15: "i8[12845056]" = torch.ops.aten.clone.default(as_strided_default_30);  as_strided_default_30 = None
        as_strided_default_31: "i8[512, 512, 7, 7]" = torch.ops.aten.as_strided.default(clone_default_15, [512, 512, 7, 7], [25088, 49, 7, 1], 0);  clone_default_15 = None
        triton_kernel_wrapper_mutation_32 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 589, grid = [(12544, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1173, 'Y_ptr': permute_139, 'Mask_prt': as_strided_default_31, 'n_elts': 12845056, 'BLOCK_SIZE': 1024});  view_1173 = triton_kernel_wrapper_mutation_32 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1176: "i8[512, 49, 512]" = torch.ops.aten.reshape.default(as_strided_default_31, [512, -1, 512]);  as_strided_default_31 = None
        view_1177: "i8[25088, 512]" = torch.ops.aten.reshape.default(view_1176, [25088, 512]);  view_1176 = None
        
        # No stacktrace found for following nodes
        as_strided_default_28: "i32[401408]" = torch.ops.aten.as_strided.default(full_default_263, [401408], [1], 0)
        clone_default_14: "i32[401408]" = torch.ops.aten.clone.default(as_strided_default_28);  as_strided_default_28 = None
        as_strided_default_29: "i32[25088, 16]" = torch.ops.aten.as_strided.default(clone_default_14, [25088, 16], [16, 1], 0);  clone_default_14 = None
        triton_kernel_wrapper_mutation_31 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 590, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1177, 'P_ptr': as_strided_default_29, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_1177 = triton_kernel_wrapper_mutation_31 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_49: "bf16[512, 512, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_290, torch.bfloat16);  primals_290 = None
        empty_428: "i32[25088, 32]" = torch.ops.aten.empty.memory_format([25088, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_429: "bf16[25088]" = torch.ops.aten.empty.memory_format([25088], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_430: "bf16[25088]" = torch.ops.aten.empty.memory_format([25088], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1180: "bf16[512, 49, 512]" = torch.ops.aten.reshape.default(permute_139, [512, -1, 512])
        view_1181: "bf16[25088, 512]" = torch.ops.aten.reshape.default(view_1180, [25088, 512]);  view_1180 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_30 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 410, constant_args_idx = 591, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1181, 'P_ptr': empty_428, 'S_ptr': empty_429, 'M_ptr': empty_430, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_1181 = triton_kernel_wrapper_mutation_30 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_48: "bf16[512, 512, 7, 7]" = torch.ops.aten.convolution.default(permute_139, convert_element_type_49, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  permute_139 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_206: "i64[]" = torch.ops.aten.add.Tensor(primals_291, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_1187: "bf16[512, 512, 49]" = torch.ops.aten.reshape.default(convolution_48, [512, 512, 49]);  convolution_48 = None
        
        # No stacktrace found for following nodes
        as_strided_default_24: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_12: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_24);  as_strided_default_24 = None
        as_strided_default_25: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_12, [512], [1], 0);  clone_default_12 = None
        as_strided_default_26: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_13: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_26);  as_strided_default_26 = None
        as_strided_default_27: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_13, [512], [1], 0);  clone_default_13 = None
        triton_kernel_wrapper_mutation_29 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 411, constant_args_idx = 592, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1187, 'SUM': as_strided_default_25, 'SUMSQ': as_strided_default_27, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_29 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_144: "f32[512]" = torch.ops.aten.div.Tensor(as_strided_default_25, full_default_260);  as_strided_default_25 = None
        div_145: "f32[512]" = torch.ops.aten.div.Tensor(as_strided_default_27, full_default_260);  as_strided_default_27 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_28 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 412, constant_args_idx = 593, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1187, 'MEAN': div_144, 'INVSTD': rsqrt_48, 'GAMMA': primals_292, 'BETA': primals_293, 'Y': permute_140, 'X_hat': permute_141, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024});  view_1187 = div_144 = primals_293 = triton_kernel_wrapper_mutation_28 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_433: "i32[25088, 32]" = torch.ops.aten.empty.memory_format([25088, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_434: "bf16[25088]" = torch.ops.aten.empty.memory_format([25088], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_435: "bf16[25088]" = torch.ops.aten.empty.memory_format([25088], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1191: "bf16[512, 49, 512]" = torch.ops.aten.reshape.default(permute_141, [512, -1, 512]);  permute_141 = None
        view_1192: "bf16[25088, 512]" = torch.ops.aten.reshape.default(view_1191, [25088, 512]);  view_1191 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_27 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 413, constant_args_idx = 594, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1192, 'P_ptr': empty_433, 'S_ptr': empty_434, 'M_ptr': empty_435, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_1192 = triton_kernel_wrapper_mutation_27 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1198: "bf16[512, 512, 7, 7]" = torch.ops.aten.reshape.default(permute_140, [512, 512, 7, 7]);  permute_140 = None
        empty_436: "bf16[512, 512, 7, 7]" = torch.ops.aten.empty.memory_format([512, 512, 7, 7], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_142: "bf16[512, 512, 7, 7]" = torch.ops.aten.permute.default(empty_436, [0, 1, 2, 3]);  empty_436 = None
        
        # No stacktrace found for following nodes
        as_strided_default_22: "i8[12845056]" = torch.ops.aten.as_strided.default(full_default_262, [12845056], [1], 0)
        clone_default_11: "i8[12845056]" = torch.ops.aten.clone.default(as_strided_default_22);  as_strided_default_22 = None
        as_strided_default_23: "i8[512, 512, 7, 7]" = torch.ops.aten.as_strided.default(clone_default_11, [512, 512, 7, 7], [25088, 49, 7, 1], 0);  clone_default_11 = None
        triton_kernel_wrapper_mutation_26 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 595, grid = [(12544, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1198, 'Y_ptr': permute_142, 'Mask_prt': as_strided_default_23, 'n_elts': 12845056, 'BLOCK_SIZE': 1024});  view_1198 = triton_kernel_wrapper_mutation_26 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1201: "i8[512, 49, 512]" = torch.ops.aten.reshape.default(as_strided_default_23, [512, -1, 512]);  as_strided_default_23 = None
        view_1202: "i8[25088, 512]" = torch.ops.aten.reshape.default(view_1201, [25088, 512]);  view_1201 = None
        
        # No stacktrace found for following nodes
        as_strided_default_20: "i32[401408]" = torch.ops.aten.as_strided.default(full_default_263, [401408], [1], 0)
        clone_default_10: "i32[401408]" = torch.ops.aten.clone.default(as_strided_default_20);  as_strided_default_20 = None
        as_strided_default_21: "i32[25088, 16]" = torch.ops.aten.as_strided.default(clone_default_10, [25088, 16], [16, 1], 0);  clone_default_10 = None
        triton_kernel_wrapper_mutation_25 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 596, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1202, 'P_ptr': as_strided_default_21, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_1202 = triton_kernel_wrapper_mutation_25 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_50: "bf16[2048, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_296, torch.bfloat16);  primals_296 = None
        empty_437: "i32[25088, 32]" = torch.ops.aten.empty.memory_format([25088, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_438: "bf16[25088]" = torch.ops.aten.empty.memory_format([25088], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_439: "bf16[25088]" = torch.ops.aten.empty.memory_format([25088], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1205: "bf16[512, 49, 512]" = torch.ops.aten.reshape.default(permute_142, [512, -1, 512])
        view_1206: "bf16[25088, 512]" = torch.ops.aten.reshape.default(view_1205, [25088, 512]);  view_1205 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_24 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 414, constant_args_idx = 597, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1206, 'P_ptr': empty_437, 'S_ptr': empty_438, 'M_ptr': empty_439, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_1206 = triton_kernel_wrapper_mutation_24 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_49: "bf16[512, 2048, 7, 7]" = torch.ops.aten.convolution.default(permute_142, convert_element_type_50, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_142 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_210: "i64[]" = torch.ops.aten.add.Tensor(primals_297, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_1212: "bf16[512, 2048, 49]" = torch.ops.aten.reshape.default(convolution_49, [512, 2048, 49]);  convolution_49 = None
        
        # No stacktrace found for following nodes
        as_strided_default_16: "f32[2048]" = torch.ops.aten.as_strided.default(full_default_264, [2048], [1], 0)
        clone_default_8: "f32[2048]" = torch.ops.aten.clone.default(as_strided_default_16);  as_strided_default_16 = None
        as_strided_default_17: "f32[2048]" = torch.ops.aten.as_strided.default(clone_default_8, [2048], [1], 0);  clone_default_8 = None
        as_strided_default_18: "f32[2048]" = torch.ops.aten.as_strided.default(full_default_264, [2048], [1], 0)
        clone_default_9: "f32[2048]" = torch.ops.aten.clone.default(as_strided_default_18);  as_strided_default_18 = None
        as_strided_default_19: "f32[2048]" = torch.ops.aten.as_strided.default(clone_default_9, [2048], [1], 0);  clone_default_9 = None
        triton_kernel_wrapper_mutation_23 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 415, constant_args_idx = 598, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1212, 'SUM': as_strided_default_17, 'SUMSQ': as_strided_default_19, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_23 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_147: "f32[2048]" = torch.ops.aten.div.Tensor(as_strided_default_17, full_default_260);  as_strided_default_17 = None
        div_148: "f32[2048]" = torch.ops.aten.div.Tensor(as_strided_default_19, full_default_260);  as_strided_default_19 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_22 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 416, constant_args_idx = 599, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1212, 'MEAN': div_147, 'INVSTD': rsqrt_49, 'GAMMA': primals_298, 'BETA': primals_299, 'Y': permute_143, 'X_hat': permute_144, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024});  view_1212 = div_147 = primals_299 = triton_kernel_wrapper_mutation_22 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_442: "i32[100352, 32]" = torch.ops.aten.empty.memory_format([100352, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_443: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_444: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1216: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(permute_144, [512, -1, 512]);  permute_144 = None
        view_1217: "bf16[100352, 512]" = torch.ops.aten.reshape.default(view_1216, [100352, 512]);  view_1216 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_21 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 417, constant_args_idx = 600, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1217, 'P_ptr': empty_442, 'S_ptr': empty_443, 'M_ptr': empty_444, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_1217 = triton_kernel_wrapper_mutation_21 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:165 in forward, code: add_14 = layer4_1_bn3 + layer4_0_relu_2;  layer4_1_bn3 = layer4_0_relu_2 = None
        view_1223: "bf16[512, 2048, 7, 7]" = torch.ops.aten.reshape.default(permute_143, [512, 2048, 7, 7]);  permute_143 = None
        add_214: "bf16[512, 2048, 7, 7]" = torch.ops.aten.add.Tensor(view_1223, permute_136);  view_1223 = permute_136 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        empty_445: "bf16[512, 2048, 7, 7]" = torch.ops.aten.empty.memory_format([512, 2048, 7, 7], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_145: "bf16[512, 2048, 7, 7]" = torch.ops.aten.permute.default(empty_445, [0, 1, 2, 3]);  empty_445 = None
        
        # No stacktrace found for following nodes
        as_strided_default_14: "i8[51380224]" = torch.ops.aten.as_strided.default(full_default_272, [51380224], [1], 0)
        clone_default_7: "i8[51380224]" = torch.ops.aten.clone.default(as_strided_default_14);  as_strided_default_14 = None
        as_strided_default_15: "i8[512, 2048, 7, 7]" = torch.ops.aten.as_strided.default(clone_default_7, [512, 2048, 7, 7], [100352, 49, 7, 1], 0);  clone_default_7 = None
        triton_kernel_wrapper_mutation_20 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 601, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_214, 'Y_ptr': permute_145, 'Mask_prt': as_strided_default_15, 'n_elts': 51380224, 'BLOCK_SIZE': 1024});  add_214 = triton_kernel_wrapper_mutation_20 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1226: "i8[512, 196, 512]" = torch.ops.aten.reshape.default(as_strided_default_15, [512, -1, 512]);  as_strided_default_15 = None
        view_1227: "i8[100352, 512]" = torch.ops.aten.reshape.default(view_1226, [100352, 512]);  view_1226 = None
        
        # No stacktrace found for following nodes
        as_strided_default_12: "i32[1605632]" = torch.ops.aten.as_strided.default(full_default_75, [1605632], [1], 0)
        clone_default_6: "i32[1605632]" = torch.ops.aten.clone.default(as_strided_default_12);  as_strided_default_12 = None
        as_strided_default_13: "i32[100352, 16]" = torch.ops.aten.as_strided.default(clone_default_6, [100352, 16], [16, 1], 0);  clone_default_6 = None
        triton_kernel_wrapper_mutation_19 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 602, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1227, 'P_ptr': as_strided_default_13, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_1227 = triton_kernel_wrapper_mutation_19 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_51: "bf16[512, 2048, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_302, torch.bfloat16);  primals_302 = None
        empty_446: "i32[100352, 32]" = torch.ops.aten.empty.memory_format([100352, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_447: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_448: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1230: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(permute_145, [512, -1, 512])
        view_1231: "bf16[100352, 512]" = torch.ops.aten.reshape.default(view_1230, [100352, 512]);  view_1230 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_18 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 418, constant_args_idx = 603, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1231, 'P_ptr': empty_446, 'S_ptr': empty_447, 'M_ptr': empty_448, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_1231 = triton_kernel_wrapper_mutation_18 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_50: "bf16[512, 512, 7, 7]" = torch.ops.aten.convolution.default(permute_145, convert_element_type_51, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_215: "i64[]" = torch.ops.aten.add.Tensor(primals_303, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_1237: "bf16[512, 512, 49]" = torch.ops.aten.reshape.default(convolution_50, [512, 512, 49]);  convolution_50 = None
        
        # No stacktrace found for following nodes
        as_strided_default_8: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_4: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_8);  as_strided_default_8 = None
        as_strided_default_9: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_4, [512], [1], 0);  clone_default_4 = None
        as_strided_default_10: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_5: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_10);  as_strided_default_10 = None
        as_strided_default_11: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_5, [512], [1], 0);  clone_default_5 = None
        triton_kernel_wrapper_mutation_17 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 419, constant_args_idx = 604, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1237, 'SUM': as_strided_default_9, 'SUMSQ': as_strided_default_11, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_17 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_150: "f32[512]" = torch.ops.aten.div.Tensor(as_strided_default_9, full_default_260);  as_strided_default_9 = None
        div_151: "f32[512]" = torch.ops.aten.div.Tensor(as_strided_default_11, full_default_260);  as_strided_default_11 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_16 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 420, constant_args_idx = 605, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1237, 'MEAN': div_150, 'INVSTD': rsqrt_50, 'GAMMA': primals_304, 'BETA': primals_305, 'Y': permute_146, 'X_hat': permute_147, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024});  view_1237 = div_150 = primals_305 = triton_kernel_wrapper_mutation_16 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_451: "i32[25088, 32]" = torch.ops.aten.empty.memory_format([25088, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_452: "bf16[25088]" = torch.ops.aten.empty.memory_format([25088], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_453: "bf16[25088]" = torch.ops.aten.empty.memory_format([25088], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1241: "bf16[512, 49, 512]" = torch.ops.aten.reshape.default(permute_147, [512, -1, 512]);  permute_147 = None
        view_1242: "bf16[25088, 512]" = torch.ops.aten.reshape.default(view_1241, [25088, 512]);  view_1241 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_15 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 421, constant_args_idx = 606, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1242, 'P_ptr': empty_451, 'S_ptr': empty_452, 'M_ptr': empty_453, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_1242 = triton_kernel_wrapper_mutation_15 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1248: "bf16[512, 512, 7, 7]" = torch.ops.aten.reshape.default(permute_146, [512, 512, 7, 7]);  permute_146 = None
        empty_454: "bf16[512, 512, 7, 7]" = torch.ops.aten.empty.memory_format([512, 512, 7, 7], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_148: "bf16[512, 512, 7, 7]" = torch.ops.aten.permute.default(empty_454, [0, 1, 2, 3]);  empty_454 = None
        
        # No stacktrace found for following nodes
        as_strided_default_6: "i8[12845056]" = torch.ops.aten.as_strided.default(full_default_262, [12845056], [1], 0)
        clone_default_3: "i8[12845056]" = torch.ops.aten.clone.default(as_strided_default_6);  as_strided_default_6 = None
        as_strided_default_7: "i8[512, 512, 7, 7]" = torch.ops.aten.as_strided.default(clone_default_3, [512, 512, 7, 7], [25088, 49, 7, 1], 0);  clone_default_3 = None
        triton_kernel_wrapper_mutation_14 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 607, grid = [(12544, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1248, 'Y_ptr': permute_148, 'Mask_prt': as_strided_default_7, 'n_elts': 12845056, 'BLOCK_SIZE': 1024});  view_1248 = triton_kernel_wrapper_mutation_14 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1251: "i8[512, 49, 512]" = torch.ops.aten.reshape.default(as_strided_default_7, [512, -1, 512]);  as_strided_default_7 = None
        view_1252: "i8[25088, 512]" = torch.ops.aten.reshape.default(view_1251, [25088, 512]);  view_1251 = None
        
        # No stacktrace found for following nodes
        as_strided_default_4: "i32[401408]" = torch.ops.aten.as_strided.default(full_default_263, [401408], [1], 0)
        clone_default_2: "i32[401408]" = torch.ops.aten.clone.default(as_strided_default_4);  as_strided_default_4 = None
        as_strided_default_5: "i32[25088, 16]" = torch.ops.aten.as_strided.default(clone_default_2, [25088, 16], [16, 1], 0);  clone_default_2 = None
        triton_kernel_wrapper_mutation_13 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 608, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1252, 'P_ptr': as_strided_default_5, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_1252 = triton_kernel_wrapper_mutation_13 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_52: "bf16[512, 512, 3, 3]" = torch.ops.prims.convert_element_type.default(primals_308, torch.bfloat16);  primals_308 = None
        empty_455: "i32[25088, 32]" = torch.ops.aten.empty.memory_format([25088, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_456: "bf16[25088]" = torch.ops.aten.empty.memory_format([25088], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_457: "bf16[25088]" = torch.ops.aten.empty.memory_format([25088], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1255: "bf16[512, 49, 512]" = torch.ops.aten.reshape.default(permute_148, [512, -1, 512])
        view_1256: "bf16[25088, 512]" = torch.ops.aten.reshape.default(view_1255, [25088, 512]);  view_1255 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_12 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 422, constant_args_idx = 609, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1256, 'P_ptr': empty_455, 'S_ptr': empty_456, 'M_ptr': empty_457, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_1256 = triton_kernel_wrapper_mutation_12 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_51: "bf16[512, 512, 7, 7]" = torch.ops.aten.convolution.default(permute_148, convert_element_type_52, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  permute_148 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_219: "i64[]" = torch.ops.aten.add.Tensor(primals_309, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_1262: "bf16[512, 512, 49]" = torch.ops.aten.reshape.default(convolution_51, [512, 512, 49]);  convolution_51 = None
        
        # No stacktrace found for following nodes
        as_strided_default_2: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_1: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_2);  as_strided_default_2 = None
        as_strided_default_3: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_1, [512], [1], 0);  clone_default_1 = None
        triton_kernel_wrapper_mutation_11 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 423, constant_args_idx = 610, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1262, 'SUM': full_default_76, 'SUMSQ': as_strided_default_3, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_11 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_153: "f32[512]" = torch.ops.aten.div.Tensor(full_default_76, full_default_260);  full_default_76 = None
        div_154: "f32[512]" = torch.ops.aten.div.Tensor(as_strided_default_3, full_default_260);  as_strided_default_3 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_10 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 424, constant_args_idx = 611, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1262, 'MEAN': div_153, 'INVSTD': rsqrt_51, 'GAMMA': primals_310, 'BETA': primals_311, 'Y': permute_149, 'X_hat': permute_150, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024});  view_1262 = div_153 = primals_311 = triton_kernel_wrapper_mutation_10 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_460: "i32[25088, 32]" = torch.ops.aten.empty.memory_format([25088, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_461: "bf16[25088]" = torch.ops.aten.empty.memory_format([25088], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_462: "bf16[25088]" = torch.ops.aten.empty.memory_format([25088], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1266: "bf16[512, 49, 512]" = torch.ops.aten.reshape.default(permute_150, [512, -1, 512]);  permute_150 = None
        view_1267: "bf16[25088, 512]" = torch.ops.aten.reshape.default(view_1266, [25088, 512]);  view_1266 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_9 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 425, constant_args_idx = 612, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1267, 'P_ptr': empty_460, 'S_ptr': empty_461, 'M_ptr': empty_462, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_1267 = triton_kernel_wrapper_mutation_9 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1273: "bf16[512, 512, 7, 7]" = torch.ops.aten.reshape.default(permute_149, [512, 512, 7, 7]);  permute_149 = None
        empty_463: "bf16[512, 512, 7, 7]" = torch.ops.aten.empty.memory_format([512, 512, 7, 7], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_151: "bf16[512, 512, 7, 7]" = torch.ops.aten.permute.default(empty_463, [0, 1, 2, 3]);  empty_463 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_8 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 613, grid = [(12544, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1273, 'Y_ptr': permute_151, 'Mask_prt': full_default_262, 'n_elts': 12845056, 'BLOCK_SIZE': 1024});  view_1273 = triton_kernel_wrapper_mutation_8 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1276: "i8[512, 49, 512]" = torch.ops.aten.reshape.default(full_default_262, [512, -1, 512]);  full_default_262 = None
        view_1277: "i8[25088, 512]" = torch.ops.aten.reshape.default(view_1276, [25088, 512]);  view_1276 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_7 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 614, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1277, 'P_ptr': full_default_263, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_1277 = triton_kernel_wrapper_mutation_7 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convert_element_type_53: "bf16[2048, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(primals_314, torch.bfloat16);  primals_314 = None
        empty_464: "i32[25088, 32]" = torch.ops.aten.empty.memory_format([25088, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_465: "bf16[25088]" = torch.ops.aten.empty.memory_format([25088], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_466: "bf16[25088]" = torch.ops.aten.empty.memory_format([25088], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1280: "bf16[512, 49, 512]" = torch.ops.aten.reshape.default(permute_151, [512, -1, 512])
        view_1281: "bf16[25088, 512]" = torch.ops.aten.reshape.default(view_1280, [25088, 512]);  view_1280 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_6 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 426, constant_args_idx = 615, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1281, 'P_ptr': empty_464, 'S_ptr': empty_465, 'M_ptr': empty_466, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_1281 = triton_kernel_wrapper_mutation_6 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        convolution_52: "bf16[512, 2048, 7, 7]" = torch.ops.aten.convolution.default(permute_151, convert_element_type_53, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  permute_151 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:136 in forward, code: self.num_batches_tracked.add_(1)
        add_223: "i64[]" = torch.ops.aten.add.Tensor(primals_315, 1)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_1287: "bf16[512, 2048, 49]" = torch.ops.aten.reshape.default(convolution_52, [512, 2048, 49]);  convolution_52 = None
        
        # No stacktrace found for following nodes
        as_strided_default: "f32[2048]" = torch.ops.aten.as_strided.default(full_default_264, [2048], [1], 0)
        clone_default: "f32[2048]" = torch.ops.aten.clone.default(as_strided_default);  as_strided_default = None
        as_strided_default_1: "f32[2048]" = torch.ops.aten.as_strided.default(clone_default, [2048], [1], 0);  clone_default = None
        triton_kernel_wrapper_mutation_5 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 427, constant_args_idx = 616, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1287, 'SUM': full_default_264, 'SUMSQ': as_strided_default_1, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_5 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        div_156: "f32[2048]" = torch.ops.aten.div.Tensor(full_default_264, full_default_260);  full_default_264 = None
        div_157: "f32[2048]" = torch.ops.aten.div.Tensor(as_strided_default_1, full_default_260);  as_strided_default_1 = full_default_260 = None
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
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_4 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 428, constant_args_idx = 617, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X': view_1287, 'MEAN': div_156, 'INVSTD': rsqrt_52, 'GAMMA': primals_316, 'BETA': primals_317, 'Y': permute_152, 'X_hat': permute_153, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024});  view_1287 = div_156 = primals_317 = triton_kernel_wrapper_mutation_4 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_469: "i32[100352, 32]" = torch.ops.aten.empty.memory_format([100352, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_470: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_471: "bf16[100352]" = torch.ops.aten.empty.memory_format([100352], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        view_1291: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(permute_153, [512, -1, 512]);  permute_153 = None
        view_1292: "bf16[100352, 512]" = torch.ops.aten.reshape.default(view_1291, [100352, 512]);  view_1291 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_3 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 429, constant_args_idx = 618, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1292, 'P_ptr': empty_469, 'S_ptr': empty_470, 'M_ptr': empty_471, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_1292 = triton_kernel_wrapper_mutation_3 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:175 in forward, code: add_15 = layer4_2_bn3 + layer4_1_relu_2;  layer4_2_bn3 = layer4_1_relu_2 = None
        view_1298: "bf16[512, 2048, 7, 7]" = torch.ops.aten.reshape.default(permute_152, [512, 2048, 7, 7]);  permute_152 = None
        add_227: "bf16[512, 2048, 7, 7]" = torch.ops.aten.add.Tensor(view_1298, permute_145);  view_1298 = permute_145 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        empty_472: "bf16[512, 2048, 7, 7]" = torch.ops.aten.empty.memory_format([512, 2048, 7, 7], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_154: "bf16[512, 2048, 7, 7]" = torch.ops.aten.permute.default(empty_472, [0, 1, 2, 3]);  empty_472 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_2 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 619, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': add_227, 'Y_ptr': permute_154, 'Mask_prt': full_default_272, 'n_elts': 51380224, 'BLOCK_SIZE': 1024});  add_227 = triton_kernel_wrapper_mutation_2 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1301: "i8[512, 196, 512]" = torch.ops.aten.reshape.default(full_default_272, [512, -1, 512]);  full_default_272 = None
        view_1302: "i8[100352, 512]" = torch.ops.aten.reshape.default(view_1301, [100352, 512]);  view_1301 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_1 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 8, constant_args_idx = 620, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1302, 'P_ptr': full_default_75, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  view_1302 = triton_kernel_wrapper_mutation_1 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:177 in forward, code: avgpool = self.avgpool(layer4_2_relu_2);  layer4_2_relu_2 = None
        mean: "bf16[512, 2048, 1, 1]" = torch.ops.aten.mean.dim(permute_154, [-1, -2], True);  permute_154 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:178 in forward, code: flatten = torch.flatten(avgpool, 1);  avgpool = None
        view_1303: "bf16[512, 2048]" = torch.ops.aten.reshape.default(mean, [512, 2048]);  mean = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:22 in forward, code: return _QuanLinear.apply(x, self.weight, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1304: "bf16[512, 4, 512]" = torch.ops.aten.reshape.default(view_1303, [512, -1, 512])
        view_1305: "bf16[2048, 512]" = torch.ops.aten.reshape.default(view_1304, [2048, 512]);  view_1304 = None
        empty_473: "i32[2048, 32]" = torch.ops.aten.empty.memory_format([2048, 32], dtype = torch.int32, device = device(type='cuda', index=0), pin_memory = False)
        empty_474: "bf16[2048]" = torch.ops.aten.empty.memory_format([2048], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        empty_475: "bf16[2048]" = torch.ops.aten.empty.memory_format([2048], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 430, constant_args_idx = 621, grid = [(2048, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X_ptr': view_1305, 'P_ptr': empty_473, 'S_ptr': empty_474, 'M_ptr': empty_475, 'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3});  view_1305 = triton_kernel_wrapper_mutation = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:22 in forward, code: return _QuanLinear.apply(x, self.weight, self.quantizer, self.target_name, self.graph_mode, self.meta)
        convert_element_type_54: "bf16[100, 2048]" = torch.ops.prims.convert_element_type.default(primals_320, torch.bfloat16);  primals_320 = None
        permute_155: "bf16[2048, 100]" = torch.ops.aten.permute.default(convert_element_type_54, [1, 0])
        mm: "bf16[512, 100]" = torch.ops.aten.mm.default(view_1303, permute_155);  view_1303 = permute_155 = None
        
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
        return (mm, primals_4, primals_10, primals_16, primals_22, primals_28, primals_34, primals_40, primals_46, primals_52, primals_58, primals_64, primals_70, primals_76, primals_82, primals_88, primals_94, primals_100, primals_106, primals_112, primals_118, primals_124, primals_130, primals_136, primals_142, primals_148, primals_154, primals_160, primals_166, primals_172, primals_178, primals_184, primals_190, primals_196, primals_202, primals_208, primals_214, primals_220, primals_226, primals_232, primals_238, primals_244, primals_250, primals_256, primals_262, primals_268, primals_274, primals_280, primals_286, primals_292, primals_298, primals_304, primals_310, primals_316, empty, empty_1, empty_2, rsqrt, empty_5, empty_6, empty_7, permute_2, as_strided_default_355, getitem_14, convert_element_type_2, empty_9, empty_10, empty_11, rsqrt_1, empty_14, empty_15, empty_16, as_strided_default_347, convert_element_type_3, empty_18, empty_19, empty_20, rsqrt_2, empty_23, empty_24, empty_25, as_strided_default_339, convert_element_type_4, empty_27, empty_28, empty_29, rsqrt_3, empty_32, empty_33, empty_34, convert_element_type_5, empty_35, empty_36, empty_37, rsqrt_4, empty_40, empty_41, empty_42, as_strided_default_327, convert_element_type_6, empty_44, empty_45, empty_46, rsqrt_5, empty_49, empty_50, empty_51, as_strided_default_319, convert_element_type_7, empty_53, empty_54, empty_55, rsqrt_6, empty_58, empty_59, empty_60, as_strided_default_311, convert_element_type_8, empty_62, empty_63, empty_64, rsqrt_7, empty_67, empty_68, empty_69, as_strided_default_303, convert_element_type_9, empty_71, empty_72, empty_73, rsqrt_8, empty_76, empty_77, empty_78, as_strided_default_295, convert_element_type_10, empty_80, empty_81, empty_82, rsqrt_9, empty_85, empty_86, empty_87, as_strided_default_291, convert_element_type_11, empty_89, empty_90, empty_91, rsqrt_10, empty_94, empty_95, empty_96, full_default_5, convert_element_type_12, empty_98, empty_99, empty_100, rsqrt_11, empty_103, empty_104, empty_105, as_strided_default_281, convert_element_type_13, empty_107, empty_108, empty_109, rsqrt_12, empty_112, empty_113, empty_114, as_strided_default_273, convert_element_type_14, empty_116, empty_117, empty_118, rsqrt_13, empty_121, empty_122, empty_123, convert_element_type_15, empty_124, empty_125, empty_126, rsqrt_14, empty_129, empty_130, empty_131, as_strided_default_261, convert_element_type_16, empty_133, empty_134, empty_135, rsqrt_15, empty_138, empty_139, empty_140, as_strided_default_253, convert_element_type_17, empty_142, empty_143, empty_144, rsqrt_16, empty_147, empty_148, empty_149, as_strided_default_245, convert_element_type_18, empty_151, empty_152, empty_153, rsqrt_17, empty_156, empty_157, empty_158, as_strided_default_237, convert_element_type_19, empty_160, empty_161, empty_162, rsqrt_18, empty_165, empty_166, empty_167, as_strided_default_229, convert_element_type_20, empty_169, empty_170, empty_171, rsqrt_19, empty_174, empty_175, empty_176, as_strided_default_221, convert_element_type_21, empty_178, empty_179, empty_180, rsqrt_20, empty_183, empty_184, empty_185, as_strided_default_213, convert_element_type_22, empty_187, empty_188, empty_189, rsqrt_21, empty_192, empty_193, empty_194, as_strided_default_205, convert_element_type_23, empty_196, empty_197, empty_198, rsqrt_22, empty_201, empty_202, empty_203, as_strided_default_201, convert_element_type_24, empty_205, empty_206, empty_207, rsqrt_23, empty_210, empty_211, empty_212, full_default_69, convert_element_type_25, empty_214, empty_215, empty_216, rsqrt_24, empty_219, empty_220, empty_221, as_strided_default_191, convert_element_type_26, empty_223, empty_224, empty_225, rsqrt_25, empty_228, empty_229, empty_230, as_strided_default_183, convert_element_type_27, empty_232, empty_233, empty_234, rsqrt_26, empty_237, empty_238, empty_239, convert_element_type_28, empty_240, empty_241, empty_242, rsqrt_27, empty_245, empty_246, empty_247, as_strided_default_171, convert_element_type_29, empty_249, empty_250, empty_251, rsqrt_28, empty_254, empty_255, empty_256, as_strided_default_163, convert_element_type_30, empty_258, empty_259, empty_260, rsqrt_29, empty_263, empty_264, empty_265, as_strided_default_155, convert_element_type_31, empty_267, empty_268, empty_269, rsqrt_30, empty_272, empty_273, empty_274, as_strided_default_147, convert_element_type_32, empty_276, empty_277, empty_278, rsqrt_31, empty_281, empty_282, empty_283, as_strided_default_139, convert_element_type_33, empty_285, empty_286, empty_287, rsqrt_32, empty_290, empty_291, empty_292, as_strided_default_131, convert_element_type_34, empty_294, empty_295, empty_296, rsqrt_33, empty_299, empty_300, empty_301, as_strided_default_123, convert_element_type_35, empty_303, empty_304, empty_305, rsqrt_34, empty_308, empty_309, empty_310, as_strided_default_115, convert_element_type_36, empty_312, empty_313, empty_314, rsqrt_35, empty_317, empty_318, empty_319, as_strided_default_107, convert_element_type_37, empty_321, empty_322, empty_323, rsqrt_36, empty_326, empty_327, empty_328, as_strided_default_99, convert_element_type_38, empty_330, empty_331, empty_332, rsqrt_37, empty_335, empty_336, empty_337, as_strided_default_91, convert_element_type_39, empty_339, empty_340, empty_341, rsqrt_38, empty_344, empty_345, empty_346, as_strided_default_83, convert_element_type_40, empty_348, empty_349, empty_350, rsqrt_39, empty_353, empty_354, empty_355, as_strided_default_75, convert_element_type_41, empty_357, empty_358, empty_359, rsqrt_40, empty_362, empty_363, empty_364, as_strided_default_67, convert_element_type_42, empty_366, empty_367, empty_368, rsqrt_41, empty_371, empty_372, empty_373, full_default_151, convert_element_type_43, empty_375, empty_376, empty_377, rsqrt_42, empty_380, empty_381, empty_382, full_default_11, convert_element_type_44, empty_384, empty_385, empty_386, rsqrt_43, empty_389, empty_390, empty_391, as_strided_default_57, convert_element_type_45, empty_393, empty_394, empty_395, rsqrt_44, empty_398, empty_399, empty_400, as_strided_default_49, convert_element_type_46, empty_402, empty_403, empty_404, rsqrt_45, empty_407, empty_408, empty_409, convert_element_type_47, empty_410, empty_411, empty_412, rsqrt_46, empty_415, empty_416, empty_417, as_strided_default_37, convert_element_type_48, empty_419, empty_420, empty_421, rsqrt_47, empty_424, empty_425, empty_426, as_strided_default_29, convert_element_type_49, empty_428, empty_429, empty_430, rsqrt_48, empty_433, empty_434, empty_435, as_strided_default_21, convert_element_type_50, empty_437, empty_438, empty_439, rsqrt_49, empty_442, empty_443, empty_444, as_strided_default_13, convert_element_type_51, empty_446, empty_447, empty_448, rsqrt_50, empty_451, empty_452, empty_453, as_strided_default_5, convert_element_type_52, empty_455, empty_456, empty_457, rsqrt_51, empty_460, empty_461, empty_462, full_default_263, convert_element_type_53, empty_464, empty_465, empty_466, rsqrt_52, empty_469, empty_470, empty_471, full_default_75, empty_473, empty_474, empty_475, convert_element_type_54)
        