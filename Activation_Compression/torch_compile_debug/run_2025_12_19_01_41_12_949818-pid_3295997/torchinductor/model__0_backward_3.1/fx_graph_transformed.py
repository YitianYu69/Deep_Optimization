class GraphModule(torch.nn.Module):
    def forward(self, primals_4: "f32[64]", primals_10: "f32[64]", primals_16: "f32[64]", primals_22: "f32[256]", primals_28: "f32[256]", primals_34: "f32[64]", primals_40: "f32[64]", primals_46: "f32[256]", primals_52: "f32[64]", primals_58: "f32[64]", primals_64: "f32[256]", primals_70: "f32[128]", primals_76: "f32[128]", primals_82: "f32[512]", primals_88: "f32[512]", primals_94: "f32[128]", primals_100: "f32[128]", primals_106: "f32[512]", primals_112: "f32[128]", primals_118: "f32[128]", primals_124: "f32[512]", primals_130: "f32[128]", primals_136: "f32[128]", primals_142: "f32[512]", primals_148: "f32[256]", primals_154: "f32[256]", primals_160: "f32[1024]", primals_166: "f32[1024]", primals_172: "f32[256]", primals_178: "f32[256]", primals_184: "f32[1024]", primals_190: "f32[256]", primals_196: "f32[256]", primals_202: "f32[1024]", primals_208: "f32[256]", primals_214: "f32[256]", primals_220: "f32[1024]", primals_226: "f32[256]", primals_232: "f32[256]", primals_238: "f32[1024]", primals_244: "f32[256]", primals_250: "f32[256]", primals_256: "f32[1024]", primals_262: "f32[512]", primals_268: "f32[512]", primals_274: "f32[2048]", primals_280: "f32[2048]", primals_286: "f32[512]", primals_292: "f32[512]", primals_298: "f32[2048]", primals_304: "f32[512]", primals_310: "f32[512]", primals_316: "f32[2048]", getitem: "i32[150528, 32]", getitem_1: "bf16[150528]", getitem_2: "bf16[150528]", rsqrt: "f32[64]", getitem_7: "i32[802816, 32]", getitem_8: "bf16[802816]", getitem_9: "bf16[802816]", getitem_10: "bf16[512, 64, 112, 112]", getitem_12: "i32[802816, 16]", getitem_14: "i8[512, 64, 56, 56]", convert_element_type_2: "bf16[64, 64, 1, 1]", getitem_15: "i32[200704, 32]", getitem_16: "bf16[200704]", getitem_17: "bf16[200704]", rsqrt_1: "f32[64]", getitem_22: "i32[200704, 32]", getitem_23: "bf16[200704]", getitem_24: "bf16[200704]", getitem_27: "i32[200704, 16]", convert_element_type_3: "bf16[64, 64, 3, 3]", getitem_28: "i32[200704, 32]", getitem_29: "bf16[200704]", getitem_30: "bf16[200704]", rsqrt_2: "f32[64]", getitem_35: "i32[200704, 32]", getitem_36: "bf16[200704]", getitem_37: "bf16[200704]", getitem_40: "i32[200704, 16]", convert_element_type_4: "bf16[256, 64, 1, 1]", getitem_41: "i32[200704, 32]", getitem_42: "bf16[200704]", getitem_43: "bf16[200704]", rsqrt_3: "f32[256]", getitem_48: "i32[802816, 32]", getitem_49: "bf16[802816]", getitem_50: "bf16[802816]", convert_element_type_5: "bf16[256, 64, 1, 1]", getitem_51: "i32[200704, 32]", getitem_52: "bf16[200704]", getitem_53: "bf16[200704]", rsqrt_4: "f32[256]", getitem_58: "i32[802816, 32]", getitem_59: "bf16[802816]", getitem_60: "bf16[802816]", getitem_63: "i32[802816, 16]", convert_element_type_6: "bf16[64, 256, 1, 1]", getitem_64: "i32[802816, 32]", getitem_65: "bf16[802816]", getitem_66: "bf16[802816]", rsqrt_5: "f32[64]", getitem_71: "i32[200704, 32]", getitem_72: "bf16[200704]", getitem_73: "bf16[200704]", getitem_76: "i32[200704, 16]", convert_element_type_7: "bf16[64, 64, 3, 3]", getitem_77: "i32[200704, 32]", getitem_78: "bf16[200704]", getitem_79: "bf16[200704]", rsqrt_6: "f32[64]", getitem_84: "i32[200704, 32]", getitem_85: "bf16[200704]", getitem_86: "bf16[200704]", getitem_89: "i32[200704, 16]", convert_element_type_8: "bf16[256, 64, 1, 1]", getitem_90: "i32[200704, 32]", getitem_91: "bf16[200704]", getitem_92: "bf16[200704]", rsqrt_7: "f32[256]", getitem_97: "i32[802816, 32]", getitem_98: "bf16[802816]", getitem_99: "bf16[802816]", getitem_102: "i32[802816, 16]", convert_element_type_9: "bf16[64, 256, 1, 1]", getitem_103: "i32[802816, 32]", getitem_104: "bf16[802816]", getitem_105: "bf16[802816]", rsqrt_8: "f32[64]", getitem_110: "i32[200704, 32]", getitem_111: "bf16[200704]", getitem_112: "bf16[200704]", getitem_115: "i32[200704, 16]", convert_element_type_10: "bf16[64, 64, 3, 3]", getitem_116: "i32[200704, 32]", getitem_117: "bf16[200704]", getitem_118: "bf16[200704]", rsqrt_9: "f32[64]", getitem_123: "i32[200704, 32]", getitem_124: "bf16[200704]", getitem_125: "bf16[200704]", getitem_128: "i32[200704, 16]", convert_element_type_11: "bf16[256, 64, 1, 1]", getitem_129: "i32[200704, 32]", getitem_130: "bf16[200704]", getitem_131: "bf16[200704]", rsqrt_10: "f32[256]", getitem_136: "i32[802816, 32]", getitem_137: "bf16[802816]", getitem_138: "bf16[802816]", getitem_141: "i32[802816, 16]", convert_element_type_12: "bf16[128, 256, 1, 1]", getitem_142: "i32[802816, 32]", getitem_143: "bf16[802816]", getitem_144: "bf16[802816]", rsqrt_11: "f32[128]", getitem_149: "i32[401408, 32]", getitem_150: "bf16[401408]", getitem_151: "bf16[401408]", getitem_154: "i32[401408, 16]", convert_element_type_13: "bf16[128, 128, 3, 3]", getitem_155: "i32[401408, 32]", getitem_156: "bf16[401408]", getitem_157: "bf16[401408]", rsqrt_12: "f32[128]", getitem_162: "i32[100352, 32]", getitem_163: "bf16[100352]", getitem_164: "bf16[100352]", getitem_167: "i32[100352, 16]", convert_element_type_14: "bf16[512, 128, 1, 1]", getitem_168: "i32[100352, 32]", getitem_169: "bf16[100352]", getitem_170: "bf16[100352]", rsqrt_13: "f32[512]", getitem_175: "i32[401408, 32]", getitem_176: "bf16[401408]", getitem_177: "bf16[401408]", convert_element_type_15: "bf16[512, 256, 1, 1]", getitem_178: "i32[802816, 32]", getitem_179: "bf16[802816]", getitem_180: "bf16[802816]", rsqrt_14: "f32[512]", getitem_185: "i32[401408, 32]", getitem_186: "bf16[401408]", getitem_187: "bf16[401408]", getitem_190: "i32[401408, 16]", convert_element_type_16: "bf16[128, 512, 1, 1]", getitem_191: "i32[401408, 32]", getitem_192: "bf16[401408]", getitem_193: "bf16[401408]", rsqrt_15: "f32[128]", getitem_198: "i32[100352, 32]", getitem_199: "bf16[100352]", getitem_200: "bf16[100352]", getitem_203: "i32[100352, 16]", convert_element_type_17: "bf16[128, 128, 3, 3]", getitem_204: "i32[100352, 32]", getitem_205: "bf16[100352]", getitem_206: "bf16[100352]", rsqrt_16: "f32[128]", getitem_211: "i32[100352, 32]", getitem_212: "bf16[100352]", getitem_213: "bf16[100352]", getitem_216: "i32[100352, 16]", convert_element_type_18: "bf16[512, 128, 1, 1]", getitem_217: "i32[100352, 32]", getitem_218: "bf16[100352]", getitem_219: "bf16[100352]", rsqrt_17: "f32[512]", getitem_224: "i32[401408, 32]", getitem_225: "bf16[401408]", getitem_226: "bf16[401408]", getitem_229: "i32[401408, 16]", convert_element_type_19: "bf16[128, 512, 1, 1]", getitem_230: "i32[401408, 32]", getitem_231: "bf16[401408]", getitem_232: "bf16[401408]", rsqrt_18: "f32[128]", getitem_237: "i32[100352, 32]", getitem_238: "bf16[100352]", getitem_239: "bf16[100352]", getitem_242: "i32[100352, 16]", convert_element_type_20: "bf16[128, 128, 3, 3]", getitem_243: "i32[100352, 32]", getitem_244: "bf16[100352]", getitem_245: "bf16[100352]", rsqrt_19: "f32[128]", getitem_250: "i32[100352, 32]", getitem_251: "bf16[100352]", getitem_252: "bf16[100352]", getitem_255: "i32[100352, 16]", convert_element_type_21: "bf16[512, 128, 1, 1]", getitem_256: "i32[100352, 32]", getitem_257: "bf16[100352]", getitem_258: "bf16[100352]", rsqrt_20: "f32[512]", getitem_263: "i32[401408, 32]", getitem_264: "bf16[401408]", getitem_265: "bf16[401408]", getitem_268: "i32[401408, 16]", convert_element_type_22: "bf16[128, 512, 1, 1]", getitem_269: "i32[401408, 32]", getitem_270: "bf16[401408]", getitem_271: "bf16[401408]", rsqrt_21: "f32[128]", getitem_276: "i32[100352, 32]", getitem_277: "bf16[100352]", getitem_278: "bf16[100352]", getitem_281: "i32[100352, 16]", convert_element_type_23: "bf16[128, 128, 3, 3]", getitem_282: "i32[100352, 32]", getitem_283: "bf16[100352]", getitem_284: "bf16[100352]", rsqrt_22: "f32[128]", getitem_289: "i32[100352, 32]", getitem_290: "bf16[100352]", getitem_291: "bf16[100352]", getitem_294: "i32[100352, 16]", convert_element_type_24: "bf16[512, 128, 1, 1]", getitem_295: "i32[100352, 32]", getitem_296: "bf16[100352]", getitem_297: "bf16[100352]", rsqrt_23: "f32[512]", getitem_302: "i32[401408, 32]", getitem_303: "bf16[401408]", getitem_304: "bf16[401408]", getitem_307: "i32[401408, 16]", convert_element_type_25: "bf16[256, 512, 1, 1]", getitem_308: "i32[401408, 32]", getitem_309: "bf16[401408]", getitem_310: "bf16[401408]", rsqrt_24: "f32[256]", getitem_315: "i32[200704, 32]", getitem_316: "bf16[200704]", getitem_317: "bf16[200704]", getitem_320: "i32[200704, 16]", convert_element_type_26: "bf16[256, 256, 3, 3]", getitem_321: "i32[200704, 32]", getitem_322: "bf16[200704]", getitem_323: "bf16[200704]", rsqrt_25: "f32[256]", getitem_328: "i32[50176, 32]", getitem_329: "bf16[50176]", getitem_330: "bf16[50176]", getitem_333: "i32[50176, 16]", convert_element_type_27: "bf16[1024, 256, 1, 1]", getitem_334: "i32[50176, 32]", getitem_335: "bf16[50176]", getitem_336: "bf16[50176]", rsqrt_26: "f32[1024]", getitem_341: "i32[200704, 32]", getitem_342: "bf16[200704]", getitem_343: "bf16[200704]", convert_element_type_28: "bf16[1024, 512, 1, 1]", getitem_344: "i32[401408, 32]", getitem_345: "bf16[401408]", getitem_346: "bf16[401408]", rsqrt_27: "f32[1024]", getitem_351: "i32[200704, 32]", getitem_352: "bf16[200704]", getitem_353: "bf16[200704]", getitem_356: "i32[200704, 16]", convert_element_type_29: "bf16[256, 1024, 1, 1]", getitem_357: "i32[200704, 32]", getitem_358: "bf16[200704]", getitem_359: "bf16[200704]", rsqrt_28: "f32[256]", getitem_364: "i32[50176, 32]", getitem_365: "bf16[50176]", getitem_366: "bf16[50176]", getitem_369: "i32[50176, 16]", convert_element_type_30: "bf16[256, 256, 3, 3]", getitem_370: "i32[50176, 32]", getitem_371: "bf16[50176]", getitem_372: "bf16[50176]", rsqrt_29: "f32[256]", getitem_377: "i32[50176, 32]", getitem_378: "bf16[50176]", getitem_379: "bf16[50176]", getitem_382: "i32[50176, 16]", convert_element_type_31: "bf16[1024, 256, 1, 1]", getitem_383: "i32[50176, 32]", getitem_384: "bf16[50176]", getitem_385: "bf16[50176]", rsqrt_30: "f32[1024]", getitem_390: "i32[200704, 32]", getitem_391: "bf16[200704]", getitem_392: "bf16[200704]", getitem_395: "i32[200704, 16]", convert_element_type_32: "bf16[256, 1024, 1, 1]", getitem_396: "i32[200704, 32]", getitem_397: "bf16[200704]", getitem_398: "bf16[200704]", rsqrt_31: "f32[256]", getitem_403: "i32[50176, 32]", getitem_404: "bf16[50176]", getitem_405: "bf16[50176]", getitem_408: "i32[50176, 16]", convert_element_type_33: "bf16[256, 256, 3, 3]", getitem_409: "i32[50176, 32]", getitem_410: "bf16[50176]", getitem_411: "bf16[50176]", rsqrt_32: "f32[256]", getitem_416: "i32[50176, 32]", getitem_417: "bf16[50176]", getitem_418: "bf16[50176]", getitem_421: "i32[50176, 16]", convert_element_type_34: "bf16[1024, 256, 1, 1]", getitem_422: "i32[50176, 32]", getitem_423: "bf16[50176]", getitem_424: "bf16[50176]", rsqrt_33: "f32[1024]", getitem_429: "i32[200704, 32]", getitem_430: "bf16[200704]", getitem_431: "bf16[200704]", getitem_434: "i32[200704, 16]", convert_element_type_35: "bf16[256, 1024, 1, 1]", getitem_435: "i32[200704, 32]", getitem_436: "bf16[200704]", getitem_437: "bf16[200704]", rsqrt_34: "f32[256]", getitem_442: "i32[50176, 32]", getitem_443: "bf16[50176]", getitem_444: "bf16[50176]", getitem_447: "i32[50176, 16]", convert_element_type_36: "bf16[256, 256, 3, 3]", getitem_448: "i32[50176, 32]", getitem_449: "bf16[50176]", getitem_450: "bf16[50176]", rsqrt_35: "f32[256]", getitem_455: "i32[50176, 32]", getitem_456: "bf16[50176]", getitem_457: "bf16[50176]", getitem_460: "i32[50176, 16]", convert_element_type_37: "bf16[1024, 256, 1, 1]", getitem_461: "i32[50176, 32]", getitem_462: "bf16[50176]", getitem_463: "bf16[50176]", rsqrt_36: "f32[1024]", getitem_468: "i32[200704, 32]", getitem_469: "bf16[200704]", getitem_470: "bf16[200704]", getitem_473: "i32[200704, 16]", convert_element_type_38: "bf16[256, 1024, 1, 1]", getitem_474: "i32[200704, 32]", getitem_475: "bf16[200704]", getitem_476: "bf16[200704]", rsqrt_37: "f32[256]", getitem_481: "i32[50176, 32]", getitem_482: "bf16[50176]", getitem_483: "bf16[50176]", getitem_486: "i32[50176, 16]", convert_element_type_39: "bf16[256, 256, 3, 3]", getitem_487: "i32[50176, 32]", getitem_488: "bf16[50176]", getitem_489: "bf16[50176]", rsqrt_38: "f32[256]", getitem_494: "i32[50176, 32]", getitem_495: "bf16[50176]", getitem_496: "bf16[50176]", getitem_499: "i32[50176, 16]", convert_element_type_40: "bf16[1024, 256, 1, 1]", getitem_500: "i32[50176, 32]", getitem_501: "bf16[50176]", getitem_502: "bf16[50176]", rsqrt_39: "f32[1024]", getitem_507: "i32[200704, 32]", getitem_508: "bf16[200704]", getitem_509: "bf16[200704]", getitem_512: "i32[200704, 16]", convert_element_type_41: "bf16[256, 1024, 1, 1]", getitem_513: "i32[200704, 32]", getitem_514: "bf16[200704]", getitem_515: "bf16[200704]", rsqrt_40: "f32[256]", getitem_520: "i32[50176, 32]", getitem_521: "bf16[50176]", getitem_522: "bf16[50176]", getitem_525: "i32[50176, 16]", convert_element_type_42: "bf16[256, 256, 3, 3]", getitem_526: "i32[50176, 32]", getitem_527: "bf16[50176]", getitem_528: "bf16[50176]", rsqrt_41: "f32[256]", getitem_533: "i32[50176, 32]", getitem_534: "bf16[50176]", getitem_535: "bf16[50176]", getitem_538: "i32[50176, 16]", convert_element_type_43: "bf16[1024, 256, 1, 1]", getitem_539: "i32[50176, 32]", getitem_540: "bf16[50176]", getitem_541: "bf16[50176]", rsqrt_42: "f32[1024]", getitem_546: "i32[200704, 32]", getitem_547: "bf16[200704]", getitem_548: "bf16[200704]", getitem_551: "i32[200704, 16]", convert_element_type_44: "bf16[512, 1024, 1, 1]", getitem_552: "i32[200704, 32]", getitem_553: "bf16[200704]", getitem_554: "bf16[200704]", rsqrt_43: "f32[512]", getitem_559: "i32[100352, 32]", getitem_560: "bf16[100352]", getitem_561: "bf16[100352]", getitem_564: "i32[100352, 16]", convert_element_type_45: "bf16[512, 512, 3, 3]", getitem_565: "i32[100352, 32]", getitem_566: "bf16[100352]", getitem_567: "bf16[100352]", rsqrt_44: "f32[512]", getitem_572: "i32[25088, 32]", getitem_573: "bf16[25088]", getitem_574: "bf16[25088]", getitem_577: "i32[25088, 16]", convert_element_type_46: "bf16[2048, 512, 1, 1]", getitem_578: "i32[25088, 32]", getitem_579: "bf16[25088]", getitem_580: "bf16[25088]", rsqrt_45: "f32[2048]", getitem_585: "i32[100352, 32]", getitem_586: "bf16[100352]", getitem_587: "bf16[100352]", convert_element_type_47: "bf16[2048, 1024, 1, 1]", getitem_588: "i32[200704, 32]", getitem_589: "bf16[200704]", getitem_590: "bf16[200704]", rsqrt_46: "f32[2048]", getitem_595: "i32[100352, 32]", getitem_596: "bf16[100352]", getitem_597: "bf16[100352]", getitem_600: "i32[100352, 16]", convert_element_type_48: "bf16[512, 2048, 1, 1]", getitem_601: "i32[100352, 32]", getitem_602: "bf16[100352]", getitem_603: "bf16[100352]", rsqrt_47: "f32[512]", getitem_608: "i32[25088, 32]", getitem_609: "bf16[25088]", getitem_610: "bf16[25088]", getitem_613: "i32[25088, 16]", convert_element_type_49: "bf16[512, 512, 3, 3]", getitem_614: "i32[25088, 32]", getitem_615: "bf16[25088]", getitem_616: "bf16[25088]", rsqrt_48: "f32[512]", getitem_621: "i32[25088, 32]", getitem_622: "bf16[25088]", getitem_623: "bf16[25088]", getitem_626: "i32[25088, 16]", convert_element_type_50: "bf16[2048, 512, 1, 1]", getitem_627: "i32[25088, 32]", getitem_628: "bf16[25088]", getitem_629: "bf16[25088]", rsqrt_49: "f32[2048]", getitem_634: "i32[100352, 32]", getitem_635: "bf16[100352]", getitem_636: "bf16[100352]", getitem_639: "i32[100352, 16]", convert_element_type_51: "bf16[512, 2048, 1, 1]", getitem_640: "i32[100352, 32]", getitem_641: "bf16[100352]", getitem_642: "bf16[100352]", rsqrt_50: "f32[512]", getitem_647: "i32[25088, 32]", getitem_648: "bf16[25088]", getitem_649: "bf16[25088]", getitem_652: "i32[25088, 16]", convert_element_type_52: "bf16[512, 512, 3, 3]", getitem_653: "i32[25088, 32]", getitem_654: "bf16[25088]", getitem_655: "bf16[25088]", rsqrt_51: "f32[512]", getitem_660: "i32[25088, 32]", getitem_661: "bf16[25088]", getitem_662: "bf16[25088]", getitem_665: "i32[25088, 16]", convert_element_type_53: "bf16[2048, 512, 1, 1]", getitem_666: "i32[25088, 32]", getitem_667: "bf16[25088]", getitem_668: "bf16[25088]", rsqrt_52: "f32[2048]", getitem_673: "i32[100352, 32]", getitem_674: "bf16[100352]", getitem_675: "bf16[100352]", getitem_678: "i32[100352, 16]", getitem_679: "i32[2048, 32]", getitem_680: "bf16[2048]", getitem_681: "bf16[2048]", convert_element_type_54: "bf16[100, 2048]", tangents_1: "bf16[512, 100]"):
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:22 in forward, code: return _QuanLinear.apply(x, self.weight, self.quantizer, self.target_name, self.graph_mode, self.meta)
        permute_156: "bf16[100, 512]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        empty_476: "bf16[2048, 512]" = torch.ops.aten.empty.memory_format([2048, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_261 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 432, constant_args_idx = 622, grid = [(2048, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_679, 'S_ptr': getitem_680, 'M_ptr': getitem_681, 'Y_ptr': empty_476, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_679 = getitem_680 = getitem_681 = triton_kernel_wrapper_mutation_261 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:22 in forward, code: return _QuanLinear.apply(x, self.weight, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1326: "bf16[512, 4, 512]" = torch.ops.aten.reshape.default(empty_476, [512, 4, 512]);  empty_476 = None
        view_1327: "bf16[512, 2048]" = torch.ops.aten.reshape.default(view_1326, [512, 2048]);  view_1326 = None
        
        # No stacktrace found for following nodes
        constant_pad_nd_default: "bf16[104, 512]" = torch.ops.aten.constant_pad_nd.default(permute_156, [0, 0, 0, 4]);  permute_156 = None
        mm_default: "bf16[104, 2048]" = torch.ops.aten.mm.default(constant_pad_nd_default, view_1327);  constant_pad_nd_default = view_1327 = None
        slice_tensor: "bf16[100, 2048]" = torch.ops.aten.slice.Tensor(mm_default, 0, 0, -4);  mm_default = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:22 in forward, code: return _QuanLinear.apply(x, self.weight, self.quantizer, self.target_name, self.graph_mode, self.meta)
        mm_2: "bf16[512, 2048]" = torch.ops.aten.mm.default(tangents_1, convert_element_type_54);  tangents_1 = convert_element_type_54 = None
        convert_element_type_62: "f32[100, 2048]" = torch.ops.prims.convert_element_type.default(slice_tensor, torch.float32);  slice_tensor = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:178 in forward, code: flatten = torch.flatten(avgpool, 1);  avgpool = None
        view_1330: "bf16[512, 2048, 1, 1]" = torch.ops.aten.reshape.default(mm_2, [512, 2048, 1, 1]);  mm_2 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:177 in forward, code: avgpool = self.avgpool(layer4_2_relu_2);  layer4_2_relu_2 = None
        expand: "bf16[512, 2048, 7, 7]" = torch.ops.aten.expand.default(view_1330, [512, 2048, 7, 7]);  view_1330 = None
        div_159: "bf16[512, 2048, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_310: "i8[100352, 512]" = torch.ops.aten.full.default([100352, 512], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        as_strided_default_284: "i8[51380224]" = torch.ops.aten.as_strided.default(full_default_310, [51380224], [1], 0)
        clone_default_142: "i8[51380224]" = torch.ops.aten.clone.default(as_strided_default_284);  as_strided_default_284 = None
        as_strided_default_285: "i8[100352, 512]" = torch.ops.aten.as_strided.default(clone_default_142, [100352, 512], [512, 1], 0);  clone_default_142 = None
        triton_kernel_wrapper_mutation_260 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 623, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_678, 'Y_ptr': as_strided_default_285, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_678 = triton_kernel_wrapper_mutation_260 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1333: "i8[512, 196, 512]" = torch.ops.aten.reshape.default(as_strided_default_285, [512, 196, 512]);  as_strided_default_285 = None
        view_1334: "i8[512, 2048, 7, 7]" = torch.ops.aten.reshape.default(view_1333, [512, 2048, 7, 7]);  view_1333 = None
        mul_371: "bf16[512, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(div_159, view_1334);  div_159 = view_1334 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_477: "bf16[100352, 512]" = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_259 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 434, constant_args_idx = 624, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_673, 'S_ptr': getitem_674, 'M_ptr': getitem_675, 'Y_ptr': empty_477, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_673 = getitem_674 = getitem_675 = triton_kernel_wrapper_mutation_259 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_1349: "bf16[512, 2048, 49]" = torch.ops.aten.reshape.default(mul_371, [512, 2048, 49])
        view_1350: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(empty_477, [512, 196, 512]);  empty_477 = None
        view_1351: "bf16[512, 2048, 49]" = torch.ops.aten.reshape.default(view_1350, [512, 2048, 49]);  view_1350 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        full_default_264: "f32[2048]" = torch.ops.aten.full.default([2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        as_strided_default_280: "f32[2048]" = torch.ops.aten.as_strided.default(full_default_264, [2048], [1], 0)
        clone_default_140: "f32[2048]" = torch.ops.aten.clone.default(as_strided_default_280);  as_strided_default_280 = None
        as_strided_default_281: "f32[2048]" = torch.ops.aten.as_strided.default(clone_default_140, [2048], [1], 0);  clone_default_140 = None
        as_strided_default_282: "f32[2048]" = torch.ops.aten.as_strided.default(full_default_264, [2048], [1], 0)
        clone_default_141: "f32[2048]" = torch.ops.aten.clone.default(as_strided_default_282);  as_strided_default_282 = None
        as_strided_default_283: "f32[2048]" = torch.ops.aten.as_strided.default(clone_default_141, [2048], [1], 0);  clone_default_141 = None
        triton_kernel_wrapper_mutation_258 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 436, constant_args_idx = 625, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1351, 'DY': view_1349, 'DBETA': as_strided_default_281, 'DGAMMA': as_strided_default_283, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_258 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_478: "bf16[512, 2048, 49]" = torch.ops.aten.empty.memory_format([512, 2048, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_157: "bf16[512, 2048, 49]" = torch.ops.aten.permute.default(empty_478, [0, 1, 2]);  empty_478 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_257 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 438, constant_args_idx = 626, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1351, 'DY': view_1349, 'INVSTD': rsqrt_52, 'GAMMA': primals_316, 'DBETA': as_strided_default_281, 'DGAMMA': as_strided_default_283, 'DX': permute_157, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024});  view_1351 = view_1349 = rsqrt_52 = primals_316 = triton_kernel_wrapper_mutation_257 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_479: "bf16[25088, 512]" = torch.ops.aten.empty.memory_format([25088, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_256 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 439, constant_args_idx = 627, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_666, 'S_ptr': getitem_667, 'M_ptr': getitem_668, 'Y_ptr': empty_479, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_666 = getitem_667 = getitem_668 = triton_kernel_wrapper_mutation_256 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_1367: "bf16[512, 2048, 7, 7]" = torch.ops.aten.reshape.default(permute_157, [512, 2048, 7, 7]);  permute_157 = None
        empty_480: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_1: "bf16[512, 512, 7, 7]" = torch.ops.aten.expand.default(empty_480, [512, 512, 7, 7]);  empty_480 = None
        convolution_backward = torch.ops.aten.convolution_backward.default(view_1367, expand_1, convert_element_type_53, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_1 = convert_element_type_53 = None
        getitem_689: "bf16[512, 512, 7, 7]" = convolution_backward[0];  convolution_backward = None
        empty_481: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_2: "bf16[2048, 512, 1, 1]" = torch.ops.aten.expand.default(empty_481, [2048, 512, 1, 1]);  empty_481 = None
        view_1368: "bf16[512, 49, 512]" = torch.ops.aten.reshape.default(empty_479, [512, 49, 512]);  empty_479 = None
        view_1369: "bf16[512, 512, 7, 7]" = torch.ops.aten.reshape.default(view_1368, [512, 512, 7, 7]);  view_1368 = None
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(view_1367, view_1369, expand_2, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1367 = view_1369 = expand_2 = None
        getitem_693: "bf16[2048, 512, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
        convert_element_type_67: "f32[2048, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_693, torch.float32);  getitem_693 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_313: "i8[25088, 512]" = torch.ops.aten.full.default([25088, 512], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        as_strided_default_278: "i8[12845056]" = torch.ops.aten.as_strided.default(full_default_313, [12845056], [1], 0)
        clone_default_139: "i8[12845056]" = torch.ops.aten.clone.default(as_strided_default_278);  as_strided_default_278 = None
        as_strided_default_279: "i8[25088, 512]" = torch.ops.aten.as_strided.default(clone_default_139, [25088, 512], [512, 1], 0);  clone_default_139 = None
        triton_kernel_wrapper_mutation_255 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 628, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_665, 'Y_ptr': as_strided_default_279, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_665 = triton_kernel_wrapper_mutation_255 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1372: "i8[512, 49, 512]" = torch.ops.aten.reshape.default(as_strided_default_279, [512, 49, 512]);  as_strided_default_279 = None
        view_1373: "i8[512, 512, 7, 7]" = torch.ops.aten.reshape.default(view_1372, [512, 512, 7, 7]);  view_1372 = None
        mul_372: "bf16[512, 512, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_689, view_1373);  getitem_689 = view_1373 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_482: "bf16[25088, 512]" = torch.ops.aten.empty.memory_format([25088, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_254 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 440, constant_args_idx = 629, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_660, 'S_ptr': getitem_661, 'M_ptr': getitem_662, 'Y_ptr': empty_482, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_660 = getitem_661 = getitem_662 = triton_kernel_wrapper_mutation_254 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_1388: "bf16[512, 512, 49]" = torch.ops.aten.reshape.default(mul_372, [512, 512, 49]);  mul_372 = None
        view_1389: "bf16[512, 49, 512]" = torch.ops.aten.reshape.default(empty_482, [512, 49, 512]);  empty_482 = None
        view_1390: "bf16[512, 512, 49]" = torch.ops.aten.reshape.default(view_1389, [512, 512, 49]);  view_1389 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        full_default_76: "f32[512]" = torch.ops.aten.full.default([512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        as_strided_default_274: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_137: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_274);  as_strided_default_274 = None
        as_strided_default_275: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_137, [512], [1], 0);  clone_default_137 = None
        as_strided_default_276: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_138: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_276);  as_strided_default_276 = None
        as_strided_default_277: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_138, [512], [1], 0);  clone_default_138 = None
        triton_kernel_wrapper_mutation_253 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 441, constant_args_idx = 630, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1390, 'DY': view_1388, 'DBETA': as_strided_default_275, 'DGAMMA': as_strided_default_277, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_253 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_483: "bf16[512, 512, 49]" = torch.ops.aten.empty.memory_format([512, 512, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_158: "bf16[512, 512, 49]" = torch.ops.aten.permute.default(empty_483, [0, 1, 2]);  empty_483 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_252 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 442, constant_args_idx = 631, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1390, 'DY': view_1388, 'INVSTD': rsqrt_51, 'GAMMA': primals_310, 'DBETA': as_strided_default_275, 'DGAMMA': as_strided_default_277, 'DX': permute_158, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024});  view_1390 = view_1388 = rsqrt_51 = primals_310 = triton_kernel_wrapper_mutation_252 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_484: "bf16[25088, 512]" = torch.ops.aten.empty.memory_format([25088, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_251 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 443, constant_args_idx = 632, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_653, 'S_ptr': getitem_654, 'M_ptr': getitem_655, 'Y_ptr': empty_484, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_653 = getitem_654 = getitem_655 = triton_kernel_wrapper_mutation_251 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_1406: "bf16[512, 512, 7, 7]" = torch.ops.aten.reshape.default(permute_158, [512, 512, 7, 7]);  permute_158 = None
        empty_485: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_3: "bf16[512, 512, 7, 7]" = torch.ops.aten.expand.default(empty_485, [512, 512, 7, 7]);  empty_485 = None
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(view_1406, expand_3, convert_element_type_52, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_3 = convert_element_type_52 = None
        getitem_701: "bf16[512, 512, 7, 7]" = convolution_backward_2[0];  convolution_backward_2 = None
        empty_486: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_4: "bf16[512, 512, 3, 3]" = torch.ops.aten.expand.default(empty_486, [512, 512, 3, 3]);  empty_486 = None
        view_1407: "bf16[512, 49, 512]" = torch.ops.aten.reshape.default(empty_484, [512, 49, 512]);  empty_484 = None
        view_1408: "bf16[512, 512, 7, 7]" = torch.ops.aten.reshape.default(view_1407, [512, 512, 7, 7]);  view_1407 = None
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(view_1406, view_1408, expand_4, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_1406 = view_1408 = expand_4 = None
        getitem_705: "bf16[512, 512, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
        convert_element_type_72: "f32[512, 512, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_705, torch.float32);  getitem_705 = None
        
        # No stacktrace found for following nodes
        as_strided_default_272: "i8[12845056]" = torch.ops.aten.as_strided.default(full_default_313, [12845056], [1], 0)
        clone_default_136: "i8[12845056]" = torch.ops.aten.clone.default(as_strided_default_272);  as_strided_default_272 = None
        as_strided_default_273: "i8[25088, 512]" = torch.ops.aten.as_strided.default(clone_default_136, [25088, 512], [512, 1], 0);  clone_default_136 = None
        triton_kernel_wrapper_mutation_250 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 633, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_652, 'Y_ptr': as_strided_default_273, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_652 = triton_kernel_wrapper_mutation_250 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1411: "i8[512, 49, 512]" = torch.ops.aten.reshape.default(as_strided_default_273, [512, 49, 512]);  as_strided_default_273 = None
        view_1412: "i8[512, 512, 7, 7]" = torch.ops.aten.reshape.default(view_1411, [512, 512, 7, 7]);  view_1411 = None
        mul_373: "bf16[512, 512, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_701, view_1412);  getitem_701 = view_1412 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_487: "bf16[25088, 512]" = torch.ops.aten.empty.memory_format([25088, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_249 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 444, constant_args_idx = 634, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_647, 'S_ptr': getitem_648, 'M_ptr': getitem_649, 'Y_ptr': empty_487, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_647 = getitem_648 = getitem_649 = triton_kernel_wrapper_mutation_249 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_1427: "bf16[512, 512, 49]" = torch.ops.aten.reshape.default(mul_373, [512, 512, 49]);  mul_373 = None
        view_1428: "bf16[512, 49, 512]" = torch.ops.aten.reshape.default(empty_487, [512, 49, 512]);  empty_487 = None
        view_1429: "bf16[512, 512, 49]" = torch.ops.aten.reshape.default(view_1428, [512, 512, 49]);  view_1428 = None
        
        # No stacktrace found for following nodes
        as_strided_default_268: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_134: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_268);  as_strided_default_268 = None
        as_strided_default_269: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_134, [512], [1], 0);  clone_default_134 = None
        as_strided_default_270: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_135: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_270);  as_strided_default_270 = None
        as_strided_default_271: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_135, [512], [1], 0);  clone_default_135 = None
        triton_kernel_wrapper_mutation_248 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 445, constant_args_idx = 635, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1429, 'DY': view_1427, 'DBETA': as_strided_default_269, 'DGAMMA': as_strided_default_271, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_248 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_488: "bf16[512, 512, 49]" = torch.ops.aten.empty.memory_format([512, 512, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_159: "bf16[512, 512, 49]" = torch.ops.aten.permute.default(empty_488, [0, 1, 2]);  empty_488 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_247 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 446, constant_args_idx = 636, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1429, 'DY': view_1427, 'INVSTD': rsqrt_50, 'GAMMA': primals_304, 'DBETA': as_strided_default_269, 'DGAMMA': as_strided_default_271, 'DX': permute_159, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024});  view_1429 = view_1427 = rsqrt_50 = primals_304 = triton_kernel_wrapper_mutation_247 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_489: "bf16[100352, 512]" = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_246 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 447, constant_args_idx = 637, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_640, 'S_ptr': getitem_641, 'M_ptr': getitem_642, 'Y_ptr': empty_489, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_640 = getitem_641 = getitem_642 = triton_kernel_wrapper_mutation_246 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_1445: "bf16[512, 512, 7, 7]" = torch.ops.aten.reshape.default(permute_159, [512, 512, 7, 7]);  permute_159 = None
        empty_490: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_5: "bf16[512, 2048, 7, 7]" = torch.ops.aten.expand.default(empty_490, [512, 2048, 7, 7]);  empty_490 = None
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(view_1445, expand_5, convert_element_type_51, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_5 = convert_element_type_51 = None
        getitem_713: "bf16[512, 2048, 7, 7]" = convolution_backward_4[0];  convolution_backward_4 = None
        empty_491: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_6: "bf16[512, 2048, 1, 1]" = torch.ops.aten.expand.default(empty_491, [512, 2048, 1, 1]);  empty_491 = None
        view_1446: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(empty_489, [512, 196, 512]);  empty_489 = None
        view_1447: "bf16[512, 2048, 7, 7]" = torch.ops.aten.reshape.default(view_1446, [512, 2048, 7, 7]);  view_1446 = None
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(view_1445, view_1447, expand_6, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1445 = view_1447 = expand_6 = None
        getitem_717: "bf16[512, 2048, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
        convert_element_type_77: "f32[512, 2048, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_717, torch.float32);  getitem_717 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_228: "bf16[512, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_371, getitem_713);  mul_371 = getitem_713 = None
        
        # No stacktrace found for following nodes
        as_strided_default_266: "i8[51380224]" = torch.ops.aten.as_strided.default(full_default_310, [51380224], [1], 0)
        clone_default_133: "i8[51380224]" = torch.ops.aten.clone.default(as_strided_default_266);  as_strided_default_266 = None
        as_strided_default_267: "i8[100352, 512]" = torch.ops.aten.as_strided.default(clone_default_133, [100352, 512], [512, 1], 0);  clone_default_133 = None
        triton_kernel_wrapper_mutation_245 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 638, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_639, 'Y_ptr': as_strided_default_267, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_639 = triton_kernel_wrapper_mutation_245 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1450: "i8[512, 196, 512]" = torch.ops.aten.reshape.default(as_strided_default_267, [512, 196, 512]);  as_strided_default_267 = None
        view_1451: "i8[512, 2048, 7, 7]" = torch.ops.aten.reshape.default(view_1450, [512, 2048, 7, 7]);  view_1450 = None
        mul_374: "bf16[512, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(add_228, view_1451);  add_228 = view_1451 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_492: "bf16[100352, 512]" = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_244 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 448, constant_args_idx = 639, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_634, 'S_ptr': getitem_635, 'M_ptr': getitem_636, 'Y_ptr': empty_492, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_634 = getitem_635 = getitem_636 = triton_kernel_wrapper_mutation_244 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_1466: "bf16[512, 2048, 49]" = torch.ops.aten.reshape.default(mul_374, [512, 2048, 49])
        view_1467: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(empty_492, [512, 196, 512]);  empty_492 = None
        view_1468: "bf16[512, 2048, 49]" = torch.ops.aten.reshape.default(view_1467, [512, 2048, 49]);  view_1467 = None
        
        # No stacktrace found for following nodes
        as_strided_default_262: "f32[2048]" = torch.ops.aten.as_strided.default(full_default_264, [2048], [1], 0)
        clone_default_131: "f32[2048]" = torch.ops.aten.clone.default(as_strided_default_262);  as_strided_default_262 = None
        as_strided_default_263: "f32[2048]" = torch.ops.aten.as_strided.default(clone_default_131, [2048], [1], 0);  clone_default_131 = None
        as_strided_default_264: "f32[2048]" = torch.ops.aten.as_strided.default(full_default_264, [2048], [1], 0)
        clone_default_132: "f32[2048]" = torch.ops.aten.clone.default(as_strided_default_264);  as_strided_default_264 = None
        as_strided_default_265: "f32[2048]" = torch.ops.aten.as_strided.default(clone_default_132, [2048], [1], 0);  clone_default_132 = None
        triton_kernel_wrapper_mutation_243 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 449, constant_args_idx = 640, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1468, 'DY': view_1466, 'DBETA': as_strided_default_263, 'DGAMMA': as_strided_default_265, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_243 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_493: "bf16[512, 2048, 49]" = torch.ops.aten.empty.memory_format([512, 2048, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_160: "bf16[512, 2048, 49]" = torch.ops.aten.permute.default(empty_493, [0, 1, 2]);  empty_493 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_242 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 450, constant_args_idx = 641, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1468, 'DY': view_1466, 'INVSTD': rsqrt_49, 'GAMMA': primals_298, 'DBETA': as_strided_default_263, 'DGAMMA': as_strided_default_265, 'DX': permute_160, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024});  view_1468 = view_1466 = rsqrt_49 = primals_298 = triton_kernel_wrapper_mutation_242 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_494: "bf16[25088, 512]" = torch.ops.aten.empty.memory_format([25088, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_241 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 451, constant_args_idx = 642, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_627, 'S_ptr': getitem_628, 'M_ptr': getitem_629, 'Y_ptr': empty_494, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_627 = getitem_628 = getitem_629 = triton_kernel_wrapper_mutation_241 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_1484: "bf16[512, 2048, 7, 7]" = torch.ops.aten.reshape.default(permute_160, [512, 2048, 7, 7]);  permute_160 = None
        empty_495: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_7: "bf16[512, 512, 7, 7]" = torch.ops.aten.expand.default(empty_495, [512, 512, 7, 7]);  empty_495 = None
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(view_1484, expand_7, convert_element_type_50, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_7 = convert_element_type_50 = None
        getitem_725: "bf16[512, 512, 7, 7]" = convolution_backward_6[0];  convolution_backward_6 = None
        empty_496: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_8: "bf16[2048, 512, 1, 1]" = torch.ops.aten.expand.default(empty_496, [2048, 512, 1, 1]);  empty_496 = None
        view_1485: "bf16[512, 49, 512]" = torch.ops.aten.reshape.default(empty_494, [512, 49, 512]);  empty_494 = None
        view_1486: "bf16[512, 512, 7, 7]" = torch.ops.aten.reshape.default(view_1485, [512, 512, 7, 7]);  view_1485 = None
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(view_1484, view_1486, expand_8, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1484 = view_1486 = expand_8 = None
        getitem_729: "bf16[2048, 512, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
        convert_element_type_82: "f32[2048, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_729, torch.float32);  getitem_729 = None
        
        # No stacktrace found for following nodes
        as_strided_default_260: "i8[12845056]" = torch.ops.aten.as_strided.default(full_default_313, [12845056], [1], 0)
        clone_default_130: "i8[12845056]" = torch.ops.aten.clone.default(as_strided_default_260);  as_strided_default_260 = None
        as_strided_default_261: "i8[25088, 512]" = torch.ops.aten.as_strided.default(clone_default_130, [25088, 512], [512, 1], 0);  clone_default_130 = None
        triton_kernel_wrapper_mutation_240 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 643, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_626, 'Y_ptr': as_strided_default_261, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_626 = triton_kernel_wrapper_mutation_240 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1489: "i8[512, 49, 512]" = torch.ops.aten.reshape.default(as_strided_default_261, [512, 49, 512]);  as_strided_default_261 = None
        view_1490: "i8[512, 512, 7, 7]" = torch.ops.aten.reshape.default(view_1489, [512, 512, 7, 7]);  view_1489 = None
        mul_375: "bf16[512, 512, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_725, view_1490);  getitem_725 = view_1490 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_497: "bf16[25088, 512]" = torch.ops.aten.empty.memory_format([25088, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_239 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 452, constant_args_idx = 644, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_621, 'S_ptr': getitem_622, 'M_ptr': getitem_623, 'Y_ptr': empty_497, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_621 = getitem_622 = getitem_623 = triton_kernel_wrapper_mutation_239 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_1505: "bf16[512, 512, 49]" = torch.ops.aten.reshape.default(mul_375, [512, 512, 49]);  mul_375 = None
        view_1506: "bf16[512, 49, 512]" = torch.ops.aten.reshape.default(empty_497, [512, 49, 512]);  empty_497 = None
        view_1507: "bf16[512, 512, 49]" = torch.ops.aten.reshape.default(view_1506, [512, 512, 49]);  view_1506 = None
        
        # No stacktrace found for following nodes
        as_strided_default_256: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_128: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_256);  as_strided_default_256 = None
        as_strided_default_257: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_128, [512], [1], 0);  clone_default_128 = None
        as_strided_default_258: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_129: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_258);  as_strided_default_258 = None
        as_strided_default_259: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_129, [512], [1], 0);  clone_default_129 = None
        triton_kernel_wrapper_mutation_238 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 453, constant_args_idx = 645, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1507, 'DY': view_1505, 'DBETA': as_strided_default_257, 'DGAMMA': as_strided_default_259, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_238 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_498: "bf16[512, 512, 49]" = torch.ops.aten.empty.memory_format([512, 512, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_161: "bf16[512, 512, 49]" = torch.ops.aten.permute.default(empty_498, [0, 1, 2]);  empty_498 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_237 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 454, constant_args_idx = 646, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1507, 'DY': view_1505, 'INVSTD': rsqrt_48, 'GAMMA': primals_292, 'DBETA': as_strided_default_257, 'DGAMMA': as_strided_default_259, 'DX': permute_161, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024});  view_1507 = view_1505 = rsqrt_48 = primals_292 = triton_kernel_wrapper_mutation_237 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_499: "bf16[25088, 512]" = torch.ops.aten.empty.memory_format([25088, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_236 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 455, constant_args_idx = 647, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_614, 'S_ptr': getitem_615, 'M_ptr': getitem_616, 'Y_ptr': empty_499, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_614 = getitem_615 = getitem_616 = triton_kernel_wrapper_mutation_236 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_1523: "bf16[512, 512, 7, 7]" = torch.ops.aten.reshape.default(permute_161, [512, 512, 7, 7]);  permute_161 = None
        empty_500: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_9: "bf16[512, 512, 7, 7]" = torch.ops.aten.expand.default(empty_500, [512, 512, 7, 7]);  empty_500 = None
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(view_1523, expand_9, convert_element_type_49, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_9 = convert_element_type_49 = None
        getitem_737: "bf16[512, 512, 7, 7]" = convolution_backward_8[0];  convolution_backward_8 = None
        empty_501: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_10: "bf16[512, 512, 3, 3]" = torch.ops.aten.expand.default(empty_501, [512, 512, 3, 3]);  empty_501 = None
        view_1524: "bf16[512, 49, 512]" = torch.ops.aten.reshape.default(empty_499, [512, 49, 512]);  empty_499 = None
        view_1525: "bf16[512, 512, 7, 7]" = torch.ops.aten.reshape.default(view_1524, [512, 512, 7, 7]);  view_1524 = None
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(view_1523, view_1525, expand_10, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_1523 = view_1525 = expand_10 = None
        getitem_741: "bf16[512, 512, 3, 3]" = convolution_backward_9[1];  convolution_backward_9 = None
        convert_element_type_87: "f32[512, 512, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_741, torch.float32);  getitem_741 = None
        
        # No stacktrace found for following nodes
        as_strided_default_254: "i8[12845056]" = torch.ops.aten.as_strided.default(full_default_313, [12845056], [1], 0)
        clone_default_127: "i8[12845056]" = torch.ops.aten.clone.default(as_strided_default_254);  as_strided_default_254 = None
        as_strided_default_255: "i8[25088, 512]" = torch.ops.aten.as_strided.default(clone_default_127, [25088, 512], [512, 1], 0);  clone_default_127 = None
        triton_kernel_wrapper_mutation_235 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 648, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_613, 'Y_ptr': as_strided_default_255, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_613 = triton_kernel_wrapper_mutation_235 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1528: "i8[512, 49, 512]" = torch.ops.aten.reshape.default(as_strided_default_255, [512, 49, 512]);  as_strided_default_255 = None
        view_1529: "i8[512, 512, 7, 7]" = torch.ops.aten.reshape.default(view_1528, [512, 512, 7, 7]);  view_1528 = None
        mul_376: "bf16[512, 512, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_737, view_1529);  getitem_737 = view_1529 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_502: "bf16[25088, 512]" = torch.ops.aten.empty.memory_format([25088, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_234 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 456, constant_args_idx = 649, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_608, 'S_ptr': getitem_609, 'M_ptr': getitem_610, 'Y_ptr': empty_502, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_608 = getitem_609 = getitem_610 = triton_kernel_wrapper_mutation_234 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_1544: "bf16[512, 512, 49]" = torch.ops.aten.reshape.default(mul_376, [512, 512, 49]);  mul_376 = None
        view_1545: "bf16[512, 49, 512]" = torch.ops.aten.reshape.default(empty_502, [512, 49, 512]);  empty_502 = None
        view_1546: "bf16[512, 512, 49]" = torch.ops.aten.reshape.default(view_1545, [512, 512, 49]);  view_1545 = None
        
        # No stacktrace found for following nodes
        as_strided_default_250: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_125: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_250);  as_strided_default_250 = None
        as_strided_default_251: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_125, [512], [1], 0);  clone_default_125 = None
        as_strided_default_252: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_126: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_252);  as_strided_default_252 = None
        as_strided_default_253: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_126, [512], [1], 0);  clone_default_126 = None
        triton_kernel_wrapper_mutation_233 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 457, constant_args_idx = 650, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1546, 'DY': view_1544, 'DBETA': as_strided_default_251, 'DGAMMA': as_strided_default_253, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_233 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_503: "bf16[512, 512, 49]" = torch.ops.aten.empty.memory_format([512, 512, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_162: "bf16[512, 512, 49]" = torch.ops.aten.permute.default(empty_503, [0, 1, 2]);  empty_503 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_232 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 458, constant_args_idx = 651, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1546, 'DY': view_1544, 'INVSTD': rsqrt_47, 'GAMMA': primals_286, 'DBETA': as_strided_default_251, 'DGAMMA': as_strided_default_253, 'DX': permute_162, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024});  view_1546 = view_1544 = rsqrt_47 = primals_286 = triton_kernel_wrapper_mutation_232 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_504: "bf16[100352, 512]" = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_231 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 459, constant_args_idx = 652, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_601, 'S_ptr': getitem_602, 'M_ptr': getitem_603, 'Y_ptr': empty_504, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_601 = getitem_602 = getitem_603 = triton_kernel_wrapper_mutation_231 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_1562: "bf16[512, 512, 7, 7]" = torch.ops.aten.reshape.default(permute_162, [512, 512, 7, 7]);  permute_162 = None
        empty_505: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_11: "bf16[512, 2048, 7, 7]" = torch.ops.aten.expand.default(empty_505, [512, 2048, 7, 7]);  empty_505 = None
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(view_1562, expand_11, convert_element_type_48, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_11 = convert_element_type_48 = None
        getitem_749: "bf16[512, 2048, 7, 7]" = convolution_backward_10[0];  convolution_backward_10 = None
        empty_506: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_12: "bf16[512, 2048, 1, 1]" = torch.ops.aten.expand.default(empty_506, [512, 2048, 1, 1]);  empty_506 = None
        view_1563: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(empty_504, [512, 196, 512]);  empty_504 = None
        view_1564: "bf16[512, 2048, 7, 7]" = torch.ops.aten.reshape.default(view_1563, [512, 2048, 7, 7]);  view_1563 = None
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(view_1562, view_1564, expand_12, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1562 = view_1564 = expand_12 = None
        getitem_753: "bf16[512, 2048, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
        convert_element_type_92: "f32[512, 2048, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_753, torch.float32);  getitem_753 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_229: "bf16[512, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_374, getitem_749);  mul_374 = getitem_749 = None
        
        # No stacktrace found for following nodes
        as_strided_default_248: "i8[51380224]" = torch.ops.aten.as_strided.default(full_default_310, [51380224], [1], 0)
        clone_default_124: "i8[51380224]" = torch.ops.aten.clone.default(as_strided_default_248);  as_strided_default_248 = None
        as_strided_default_249: "i8[100352, 512]" = torch.ops.aten.as_strided.default(clone_default_124, [100352, 512], [512, 1], 0);  clone_default_124 = None
        triton_kernel_wrapper_mutation_230 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 653, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_600, 'Y_ptr': as_strided_default_249, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_600 = triton_kernel_wrapper_mutation_230 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1567: "i8[512, 196, 512]" = torch.ops.aten.reshape.default(as_strided_default_249, [512, 196, 512]);  as_strided_default_249 = None
        view_1568: "i8[512, 2048, 7, 7]" = torch.ops.aten.reshape.default(view_1567, [512, 2048, 7, 7]);  view_1567 = None
        mul_377: "bf16[512, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(add_229, view_1568);  add_229 = view_1568 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_507: "bf16[100352, 512]" = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_229 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 460, constant_args_idx = 654, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_595, 'S_ptr': getitem_596, 'M_ptr': getitem_597, 'Y_ptr': empty_507, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_595 = getitem_596 = getitem_597 = triton_kernel_wrapper_mutation_229 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_1583: "bf16[512, 2048, 49]" = torch.ops.aten.reshape.default(mul_377, [512, 2048, 49]);  mul_377 = None
        view_1584: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(empty_507, [512, 196, 512]);  empty_507 = None
        view_1585: "bf16[512, 2048, 49]" = torch.ops.aten.reshape.default(view_1584, [512, 2048, 49]);  view_1584 = None
        
        # No stacktrace found for following nodes
        as_strided_default_244: "f32[2048]" = torch.ops.aten.as_strided.default(full_default_264, [2048], [1], 0)
        clone_default_122: "f32[2048]" = torch.ops.aten.clone.default(as_strided_default_244);  as_strided_default_244 = None
        as_strided_default_245: "f32[2048]" = torch.ops.aten.as_strided.default(clone_default_122, [2048], [1], 0);  clone_default_122 = None
        as_strided_default_246: "f32[2048]" = torch.ops.aten.as_strided.default(full_default_264, [2048], [1], 0)
        clone_default_123: "f32[2048]" = torch.ops.aten.clone.default(as_strided_default_246);  as_strided_default_246 = None
        as_strided_default_247: "f32[2048]" = torch.ops.aten.as_strided.default(clone_default_123, [2048], [1], 0);  clone_default_123 = None
        triton_kernel_wrapper_mutation_228 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 461, constant_args_idx = 655, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1585, 'DY': view_1583, 'DBETA': as_strided_default_245, 'DGAMMA': as_strided_default_247, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_228 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_508: "bf16[512, 2048, 49]" = torch.ops.aten.empty.memory_format([512, 2048, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_163: "bf16[512, 2048, 49]" = torch.ops.aten.permute.default(empty_508, [0, 1, 2]);  empty_508 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_227 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 462, constant_args_idx = 656, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1585, 'DY': view_1583, 'INVSTD': rsqrt_46, 'GAMMA': primals_280, 'DBETA': as_strided_default_245, 'DGAMMA': as_strided_default_247, 'DX': permute_163, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024});  view_1585 = rsqrt_46 = primals_280 = triton_kernel_wrapper_mutation_227 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_509: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_226 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 463, constant_args_idx = 657, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_588, 'S_ptr': getitem_589, 'M_ptr': getitem_590, 'Y_ptr': empty_509, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_588 = getitem_589 = getitem_590 = triton_kernel_wrapper_mutation_226 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_1601: "bf16[512, 2048, 7, 7]" = torch.ops.aten.reshape.default(permute_163, [512, 2048, 7, 7]);  permute_163 = None
        empty_510: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_13: "bf16[512, 1024, 14, 14]" = torch.ops.aten.expand.default(empty_510, [512, 1024, 14, 14]);  empty_510 = None
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(view_1601, expand_13, convert_element_type_47, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_13 = convert_element_type_47 = None
        getitem_761: "bf16[512, 1024, 14, 14]" = convolution_backward_12[0];  convolution_backward_12 = None
        empty_511: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_14: "bf16[2048, 1024, 1, 1]" = torch.ops.aten.expand.default(empty_511, [2048, 1024, 1, 1]);  empty_511 = None
        view_1602: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_509, [512, 392, 512]);  empty_509 = None
        view_1603: "bf16[512, 1024, 14, 14]" = torch.ops.aten.reshape.default(view_1602, [512, 1024, 14, 14]);  view_1602 = None
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(view_1601, view_1603, expand_14, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1601 = view_1603 = expand_14 = None
        getitem_765: "bf16[2048, 1024, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
        convert_element_type_97: "f32[2048, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_765, torch.float32);  getitem_765 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_512: "bf16[100352, 512]" = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_225 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 464, constant_args_idx = 658, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_585, 'S_ptr': getitem_586, 'M_ptr': getitem_587, 'Y_ptr': empty_512, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_585 = getitem_586 = getitem_587 = triton_kernel_wrapper_mutation_225 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_1619: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(empty_512, [512, 196, 512]);  empty_512 = None
        view_1620: "bf16[512, 2048, 49]" = torch.ops.aten.reshape.default(view_1619, [512, 2048, 49]);  view_1619 = None
        
        # No stacktrace found for following nodes
        as_strided_default_242: "f32[2048]" = torch.ops.aten.as_strided.default(full_default_264, [2048], [1], 0)
        clone_default_121: "f32[2048]" = torch.ops.aten.clone.default(as_strided_default_242);  as_strided_default_242 = None
        as_strided_default_243: "f32[2048]" = torch.ops.aten.as_strided.default(clone_default_121, [2048], [1], 0);  clone_default_121 = None
        triton_kernel_wrapper_mutation_224 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 465, constant_args_idx = 659, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1620, 'DY': view_1583, 'DBETA': full_default_264, 'DGAMMA': as_strided_default_243, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_224 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_513: "bf16[512, 2048, 49]" = torch.ops.aten.empty.memory_format([512, 2048, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_164: "bf16[512, 2048, 49]" = torch.ops.aten.permute.default(empty_513, [0, 1, 2]);  empty_513 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_223 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 466, constant_args_idx = 660, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1620, 'DY': view_1583, 'INVSTD': rsqrt_45, 'GAMMA': primals_274, 'DBETA': full_default_264, 'DGAMMA': as_strided_default_243, 'DX': permute_164, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024});  view_1620 = view_1583 = rsqrt_45 = primals_274 = triton_kernel_wrapper_mutation_223 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_514: "bf16[25088, 512]" = torch.ops.aten.empty.memory_format([25088, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_222 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 467, constant_args_idx = 661, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_578, 'S_ptr': getitem_579, 'M_ptr': getitem_580, 'Y_ptr': empty_514, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_578 = getitem_579 = getitem_580 = triton_kernel_wrapper_mutation_222 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_1636: "bf16[512, 2048, 7, 7]" = torch.ops.aten.reshape.default(permute_164, [512, 2048, 7, 7]);  permute_164 = None
        empty_515: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_15: "bf16[512, 512, 7, 7]" = torch.ops.aten.expand.default(empty_515, [512, 512, 7, 7]);  empty_515 = None
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(view_1636, expand_15, convert_element_type_46, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_15 = convert_element_type_46 = None
        getitem_772: "bf16[512, 512, 7, 7]" = convolution_backward_14[0];  convolution_backward_14 = None
        empty_516: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_16: "bf16[2048, 512, 1, 1]" = torch.ops.aten.expand.default(empty_516, [2048, 512, 1, 1]);  empty_516 = None
        view_1637: "bf16[512, 49, 512]" = torch.ops.aten.reshape.default(empty_514, [512, 49, 512]);  empty_514 = None
        view_1638: "bf16[512, 512, 7, 7]" = torch.ops.aten.reshape.default(view_1637, [512, 512, 7, 7]);  view_1637 = None
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(view_1636, view_1638, expand_16, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1636 = view_1638 = expand_16 = None
        getitem_776: "bf16[2048, 512, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
        convert_element_type_102: "f32[2048, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_776, torch.float32);  getitem_776 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_221 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 662, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_577, 'Y_ptr': full_default_313, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_577 = triton_kernel_wrapper_mutation_221 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1641: "i8[512, 49, 512]" = torch.ops.aten.reshape.default(full_default_313, [512, 49, 512]);  full_default_313 = None
        view_1642: "i8[512, 512, 7, 7]" = torch.ops.aten.reshape.default(view_1641, [512, 512, 7, 7]);  view_1641 = None
        mul_378: "bf16[512, 512, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_772, view_1642);  getitem_772 = view_1642 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_517: "bf16[25088, 512]" = torch.ops.aten.empty.memory_format([25088, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_220 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 468, constant_args_idx = 663, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_572, 'S_ptr': getitem_573, 'M_ptr': getitem_574, 'Y_ptr': empty_517, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_572 = getitem_573 = getitem_574 = triton_kernel_wrapper_mutation_220 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_1657: "bf16[512, 512, 49]" = torch.ops.aten.reshape.default(mul_378, [512, 512, 49]);  mul_378 = None
        view_1658: "bf16[512, 49, 512]" = torch.ops.aten.reshape.default(empty_517, [512, 49, 512]);  empty_517 = None
        view_1659: "bf16[512, 512, 49]" = torch.ops.aten.reshape.default(view_1658, [512, 512, 49]);  view_1658 = None
        
        # No stacktrace found for following nodes
        as_strided_default_238: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_119: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_238);  as_strided_default_238 = None
        as_strided_default_239: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_119, [512], [1], 0);  clone_default_119 = None
        as_strided_default_240: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_120: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_240);  as_strided_default_240 = None
        as_strided_default_241: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_120, [512], [1], 0);  clone_default_120 = None
        triton_kernel_wrapper_mutation_219 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 469, constant_args_idx = 664, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1659, 'DY': view_1657, 'DBETA': as_strided_default_239, 'DGAMMA': as_strided_default_241, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_219 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_518: "bf16[512, 512, 49]" = torch.ops.aten.empty.memory_format([512, 512, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_165: "bf16[512, 512, 49]" = torch.ops.aten.permute.default(empty_518, [0, 1, 2]);  empty_518 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_218 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 470, constant_args_idx = 665, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1659, 'DY': view_1657, 'INVSTD': rsqrt_44, 'GAMMA': primals_268, 'DBETA': as_strided_default_239, 'DGAMMA': as_strided_default_241, 'DX': permute_165, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024});  view_1659 = view_1657 = rsqrt_44 = primals_268 = triton_kernel_wrapper_mutation_218 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_519: "bf16[100352, 512]" = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_217 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 471, constant_args_idx = 666, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_565, 'S_ptr': getitem_566, 'M_ptr': getitem_567, 'Y_ptr': empty_519, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_565 = getitem_566 = getitem_567 = triton_kernel_wrapper_mutation_217 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_1675: "bf16[512, 512, 7, 7]" = torch.ops.aten.reshape.default(permute_165, [512, 512, 7, 7]);  permute_165 = None
        empty_520: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_17: "bf16[512, 512, 14, 14]" = torch.ops.aten.expand.default(empty_520, [512, 512, 14, 14]);  empty_520 = None
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(view_1675, expand_17, convert_element_type_45, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_17 = convert_element_type_45 = None
        getitem_784: "bf16[512, 512, 14, 14]" = convolution_backward_16[0];  convolution_backward_16 = None
        empty_521: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_18: "bf16[512, 512, 3, 3]" = torch.ops.aten.expand.default(empty_521, [512, 512, 3, 3]);  empty_521 = None
        view_1676: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(empty_519, [512, 196, 512]);  empty_519 = None
        view_1677: "bf16[512, 512, 14, 14]" = torch.ops.aten.reshape.default(view_1676, [512, 512, 14, 14]);  view_1676 = None
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(view_1675, view_1677, expand_18, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_1675 = view_1677 = expand_18 = None
        getitem_788: "bf16[512, 512, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None
        convert_element_type_107: "f32[512, 512, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_788, torch.float32);  getitem_788 = None
        
        # No stacktrace found for following nodes
        as_strided_default_236: "i8[51380224]" = torch.ops.aten.as_strided.default(full_default_310, [51380224], [1], 0)
        clone_default_118: "i8[51380224]" = torch.ops.aten.clone.default(as_strided_default_236);  as_strided_default_236 = None
        as_strided_default_237: "i8[100352, 512]" = torch.ops.aten.as_strided.default(clone_default_118, [100352, 512], [512, 1], 0);  clone_default_118 = None
        triton_kernel_wrapper_mutation_216 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 667, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_564, 'Y_ptr': as_strided_default_237, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_564 = triton_kernel_wrapper_mutation_216 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1680: "i8[512, 196, 512]" = torch.ops.aten.reshape.default(as_strided_default_237, [512, 196, 512]);  as_strided_default_237 = None
        view_1681: "i8[512, 512, 14, 14]" = torch.ops.aten.reshape.default(view_1680, [512, 512, 14, 14]);  view_1680 = None
        mul_379: "bf16[512, 512, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_784, view_1681);  getitem_784 = view_1681 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_522: "bf16[100352, 512]" = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_215 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 472, constant_args_idx = 668, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_559, 'S_ptr': getitem_560, 'M_ptr': getitem_561, 'Y_ptr': empty_522, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_559 = getitem_560 = getitem_561 = triton_kernel_wrapper_mutation_215 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_1696: "bf16[512, 512, 196]" = torch.ops.aten.reshape.default(mul_379, [512, 512, 196]);  mul_379 = None
        view_1697: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(empty_522, [512, 196, 512]);  empty_522 = None
        view_1698: "bf16[512, 512, 196]" = torch.ops.aten.reshape.default(view_1697, [512, 512, 196]);  view_1697 = None
        
        # No stacktrace found for following nodes
        as_strided_default_232: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_116: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_232);  as_strided_default_232 = None
        as_strided_default_233: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_116, [512], [1], 0);  clone_default_116 = None
        as_strided_default_234: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_117: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_234);  as_strided_default_234 = None
        as_strided_default_235: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_117, [512], [1], 0);  clone_default_117 = None
        triton_kernel_wrapper_mutation_214 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 473, constant_args_idx = 669, grid = [(512, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1698, 'DY': view_1696, 'DBETA': as_strided_default_233, 'DGAMMA': as_strided_default_235, 'M': 100352, 'HW': 196, 'stride_n': 100352, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_214 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_523: "bf16[512, 512, 196]" = torch.ops.aten.empty.memory_format([512, 512, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_166: "bf16[512, 512, 196]" = torch.ops.aten.permute.default(empty_523, [0, 1, 2]);  empty_523 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_213 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 474, constant_args_idx = 670, grid = [(512, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1698, 'DY': view_1696, 'INVSTD': rsqrt_43, 'GAMMA': primals_262, 'DBETA': as_strided_default_233, 'DGAMMA': as_strided_default_235, 'DX': permute_166, 'M': 100352, 'HW': 196, 'stride_n': 100352, 'stride_c': 196, 'BLOCK_M': 1024});  view_1698 = view_1696 = rsqrt_43 = primals_262 = triton_kernel_wrapper_mutation_213 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_524: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_212 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 475, constant_args_idx = 671, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_552, 'S_ptr': getitem_553, 'M_ptr': getitem_554, 'Y_ptr': empty_524, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_552 = getitem_553 = getitem_554 = triton_kernel_wrapper_mutation_212 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_1714: "bf16[512, 512, 14, 14]" = torch.ops.aten.reshape.default(permute_166, [512, 512, 14, 14]);  permute_166 = None
        empty_525: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_19: "bf16[512, 1024, 14, 14]" = torch.ops.aten.expand.default(empty_525, [512, 1024, 14, 14]);  empty_525 = None
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(view_1714, expand_19, convert_element_type_44, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_19 = convert_element_type_44 = None
        getitem_796: "bf16[512, 1024, 14, 14]" = convolution_backward_18[0];  convolution_backward_18 = None
        empty_526: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_20: "bf16[512, 1024, 1, 1]" = torch.ops.aten.expand.default(empty_526, [512, 1024, 1, 1]);  empty_526 = None
        view_1715: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_524, [512, 392, 512]);  empty_524 = None
        view_1716: "bf16[512, 1024, 14, 14]" = torch.ops.aten.reshape.default(view_1715, [512, 1024, 14, 14]);  view_1715 = None
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(view_1714, view_1716, expand_20, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1714 = view_1716 = expand_20 = None
        getitem_800: "bf16[512, 1024, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
        convert_element_type_112: "f32[512, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_800, torch.float32);  getitem_800 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_230: "bf16[512, 1024, 14, 14]" = torch.ops.aten.add.Tensor(getitem_761, getitem_796);  getitem_761 = getitem_796 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_339: "i8[200704, 512]" = torch.ops.aten.full.default([200704, 512], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        as_strided_default_230: "i8[102760448]" = torch.ops.aten.as_strided.default(full_default_339, [102760448], [1], 0)
        clone_default_115: "i8[102760448]" = torch.ops.aten.clone.default(as_strided_default_230);  as_strided_default_230 = None
        as_strided_default_231: "i8[200704, 512]" = torch.ops.aten.as_strided.default(clone_default_115, [200704, 512], [512, 1], 0);  clone_default_115 = None
        triton_kernel_wrapper_mutation_211 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 672, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_551, 'Y_ptr': as_strided_default_231, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_551 = triton_kernel_wrapper_mutation_211 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1719: "i8[512, 392, 512]" = torch.ops.aten.reshape.default(as_strided_default_231, [512, 392, 512]);  as_strided_default_231 = None
        view_1720: "i8[512, 1024, 14, 14]" = torch.ops.aten.reshape.default(view_1719, [512, 1024, 14, 14]);  view_1719 = None
        mul_380: "bf16[512, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(add_230, view_1720);  add_230 = view_1720 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_527: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_210 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 476, constant_args_idx = 673, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_546, 'S_ptr': getitem_547, 'M_ptr': getitem_548, 'Y_ptr': empty_527, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_546 = getitem_547 = getitem_548 = triton_kernel_wrapper_mutation_210 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_1735: "bf16[512, 1024, 196]" = torch.ops.aten.reshape.default(mul_380, [512, 1024, 196])
        view_1736: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_527, [512, 392, 512]);  empty_527 = None
        view_1737: "bf16[512, 1024, 196]" = torch.ops.aten.reshape.default(view_1736, [512, 1024, 196]);  view_1736 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        full_default_152: "f32[1024]" = torch.ops.aten.full.default([1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        as_strided_default_226: "f32[1024]" = torch.ops.aten.as_strided.default(full_default_152, [1024], [1], 0)
        clone_default_113: "f32[1024]" = torch.ops.aten.clone.default(as_strided_default_226);  as_strided_default_226 = None
        as_strided_default_227: "f32[1024]" = torch.ops.aten.as_strided.default(clone_default_113, [1024], [1], 0);  clone_default_113 = None
        as_strided_default_228: "f32[1024]" = torch.ops.aten.as_strided.default(full_default_152, [1024], [1], 0)
        clone_default_114: "f32[1024]" = torch.ops.aten.clone.default(as_strided_default_228);  as_strided_default_228 = None
        as_strided_default_229: "f32[1024]" = torch.ops.aten.as_strided.default(clone_default_114, [1024], [1], 0);  clone_default_114 = None
        triton_kernel_wrapper_mutation_209 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 477, constant_args_idx = 674, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1737, 'DY': view_1735, 'DBETA': as_strided_default_227, 'DGAMMA': as_strided_default_229, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_209 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_528: "bf16[512, 1024, 196]" = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_167: "bf16[512, 1024, 196]" = torch.ops.aten.permute.default(empty_528, [0, 1, 2]);  empty_528 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_208 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 478, constant_args_idx = 675, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1737, 'DY': view_1735, 'INVSTD': rsqrt_42, 'GAMMA': primals_256, 'DBETA': as_strided_default_227, 'DGAMMA': as_strided_default_229, 'DX': permute_167, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024});  view_1737 = view_1735 = rsqrt_42 = primals_256 = triton_kernel_wrapper_mutation_208 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_529: "bf16[50176, 512]" = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_207 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 479, constant_args_idx = 676, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_539, 'S_ptr': getitem_540, 'M_ptr': getitem_541, 'Y_ptr': empty_529, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_539 = getitem_540 = getitem_541 = triton_kernel_wrapper_mutation_207 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_1753: "bf16[512, 1024, 14, 14]" = torch.ops.aten.reshape.default(permute_167, [512, 1024, 14, 14]);  permute_167 = None
        empty_530: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_21: "bf16[512, 256, 14, 14]" = torch.ops.aten.expand.default(empty_530, [512, 256, 14, 14]);  empty_530 = None
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(view_1753, expand_21, convert_element_type_43, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_21 = convert_element_type_43 = None
        getitem_808: "bf16[512, 256, 14, 14]" = convolution_backward_20[0];  convolution_backward_20 = None
        empty_531: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_22: "bf16[1024, 256, 1, 1]" = torch.ops.aten.expand.default(empty_531, [1024, 256, 1, 1]);  empty_531 = None
        view_1754: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(empty_529, [512, 98, 512]);  empty_529 = None
        view_1755: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(view_1754, [512, 256, 14, 14]);  view_1754 = None
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(view_1753, view_1755, expand_22, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1753 = view_1755 = expand_22 = None
        getitem_812: "bf16[1024, 256, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
        convert_element_type_117: "f32[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_812, torch.float32);  getitem_812 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_342: "i8[50176, 512]" = torch.ops.aten.full.default([50176, 512], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        as_strided_default_224: "i8[25690112]" = torch.ops.aten.as_strided.default(full_default_342, [25690112], [1], 0)
        clone_default_112: "i8[25690112]" = torch.ops.aten.clone.default(as_strided_default_224);  as_strided_default_224 = None
        as_strided_default_225: "i8[50176, 512]" = torch.ops.aten.as_strided.default(clone_default_112, [50176, 512], [512, 1], 0);  clone_default_112 = None
        triton_kernel_wrapper_mutation_206 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 677, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_538, 'Y_ptr': as_strided_default_225, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_538 = triton_kernel_wrapper_mutation_206 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1758: "i8[512, 98, 512]" = torch.ops.aten.reshape.default(as_strided_default_225, [512, 98, 512]);  as_strided_default_225 = None
        view_1759: "i8[512, 256, 14, 14]" = torch.ops.aten.reshape.default(view_1758, [512, 256, 14, 14]);  view_1758 = None
        mul_381: "bf16[512, 256, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_808, view_1759);  getitem_808 = view_1759 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_532: "bf16[50176, 512]" = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_205 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 480, constant_args_idx = 678, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_533, 'S_ptr': getitem_534, 'M_ptr': getitem_535, 'Y_ptr': empty_532, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_533 = getitem_534 = getitem_535 = triton_kernel_wrapper_mutation_205 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_1774: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(mul_381, [512, 256, 196]);  mul_381 = None
        view_1775: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(empty_532, [512, 98, 512]);  empty_532 = None
        view_1776: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(view_1775, [512, 256, 196]);  view_1775 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        full_default_18: "f32[256]" = torch.ops.aten.full.default([256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        as_strided_default_220: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_110: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_220);  as_strided_default_220 = None
        as_strided_default_221: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_110, [256], [1], 0);  clone_default_110 = None
        as_strided_default_222: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_111: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_222);  as_strided_default_222 = None
        as_strided_default_223: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_111, [256], [1], 0);  clone_default_111 = None
        triton_kernel_wrapper_mutation_204 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 481, constant_args_idx = 679, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1776, 'DY': view_1774, 'DBETA': as_strided_default_221, 'DGAMMA': as_strided_default_223, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_204 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_533: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_168: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_533, [0, 1, 2]);  empty_533 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_203 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 482, constant_args_idx = 680, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1776, 'DY': view_1774, 'INVSTD': rsqrt_41, 'GAMMA': primals_250, 'DBETA': as_strided_default_221, 'DGAMMA': as_strided_default_223, 'DX': permute_168, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  view_1776 = view_1774 = rsqrt_41 = primals_250 = triton_kernel_wrapper_mutation_203 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_534: "bf16[50176, 512]" = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_202 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 483, constant_args_idx = 681, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_526, 'S_ptr': getitem_527, 'M_ptr': getitem_528, 'Y_ptr': empty_534, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_526 = getitem_527 = getitem_528 = triton_kernel_wrapper_mutation_202 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_1792: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(permute_168, [512, 256, 14, 14]);  permute_168 = None
        empty_535: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_23: "bf16[512, 256, 14, 14]" = torch.ops.aten.expand.default(empty_535, [512, 256, 14, 14]);  empty_535 = None
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(view_1792, expand_23, convert_element_type_42, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_23 = convert_element_type_42 = None
        getitem_820: "bf16[512, 256, 14, 14]" = convolution_backward_22[0];  convolution_backward_22 = None
        empty_536: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_24: "bf16[256, 256, 3, 3]" = torch.ops.aten.expand.default(empty_536, [256, 256, 3, 3]);  empty_536 = None
        view_1793: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(empty_534, [512, 98, 512]);  empty_534 = None
        view_1794: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(view_1793, [512, 256, 14, 14]);  view_1793 = None
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(view_1792, view_1794, expand_24, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_1792 = view_1794 = expand_24 = None
        getitem_824: "bf16[256, 256, 3, 3]" = convolution_backward_23[1];  convolution_backward_23 = None
        convert_element_type_122: "f32[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_824, torch.float32);  getitem_824 = None
        
        # No stacktrace found for following nodes
        as_strided_default_218: "i8[25690112]" = torch.ops.aten.as_strided.default(full_default_342, [25690112], [1], 0)
        clone_default_109: "i8[25690112]" = torch.ops.aten.clone.default(as_strided_default_218);  as_strided_default_218 = None
        as_strided_default_219: "i8[50176, 512]" = torch.ops.aten.as_strided.default(clone_default_109, [50176, 512], [512, 1], 0);  clone_default_109 = None
        triton_kernel_wrapper_mutation_201 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 682, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_525, 'Y_ptr': as_strided_default_219, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_525 = triton_kernel_wrapper_mutation_201 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1797: "i8[512, 98, 512]" = torch.ops.aten.reshape.default(as_strided_default_219, [512, 98, 512]);  as_strided_default_219 = None
        view_1798: "i8[512, 256, 14, 14]" = torch.ops.aten.reshape.default(view_1797, [512, 256, 14, 14]);  view_1797 = None
        mul_382: "bf16[512, 256, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_820, view_1798);  getitem_820 = view_1798 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_537: "bf16[50176, 512]" = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_200 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 484, constant_args_idx = 683, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_520, 'S_ptr': getitem_521, 'M_ptr': getitem_522, 'Y_ptr': empty_537, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_520 = getitem_521 = getitem_522 = triton_kernel_wrapper_mutation_200 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_1813: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(mul_382, [512, 256, 196]);  mul_382 = None
        view_1814: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(empty_537, [512, 98, 512]);  empty_537 = None
        view_1815: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(view_1814, [512, 256, 196]);  view_1814 = None
        
        # No stacktrace found for following nodes
        as_strided_default_214: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_107: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_214);  as_strided_default_214 = None
        as_strided_default_215: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_107, [256], [1], 0);  clone_default_107 = None
        as_strided_default_216: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_108: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_216);  as_strided_default_216 = None
        as_strided_default_217: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_108, [256], [1], 0);  clone_default_108 = None
        triton_kernel_wrapper_mutation_199 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 485, constant_args_idx = 684, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1815, 'DY': view_1813, 'DBETA': as_strided_default_215, 'DGAMMA': as_strided_default_217, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_199 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_538: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_169: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_538, [0, 1, 2]);  empty_538 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_198 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 486, constant_args_idx = 685, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1815, 'DY': view_1813, 'INVSTD': rsqrt_40, 'GAMMA': primals_244, 'DBETA': as_strided_default_215, 'DGAMMA': as_strided_default_217, 'DX': permute_169, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  view_1815 = view_1813 = rsqrt_40 = primals_244 = triton_kernel_wrapper_mutation_198 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_539: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_197 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 487, constant_args_idx = 686, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_513, 'S_ptr': getitem_514, 'M_ptr': getitem_515, 'Y_ptr': empty_539, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_513 = getitem_514 = getitem_515 = triton_kernel_wrapper_mutation_197 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_1831: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(permute_169, [512, 256, 14, 14]);  permute_169 = None
        empty_540: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_25: "bf16[512, 1024, 14, 14]" = torch.ops.aten.expand.default(empty_540, [512, 1024, 14, 14]);  empty_540 = None
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(view_1831, expand_25, convert_element_type_41, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_25 = convert_element_type_41 = None
        getitem_832: "bf16[512, 1024, 14, 14]" = convolution_backward_24[0];  convolution_backward_24 = None
        empty_541: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_26: "bf16[256, 1024, 1, 1]" = torch.ops.aten.expand.default(empty_541, [256, 1024, 1, 1]);  empty_541 = None
        view_1832: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_539, [512, 392, 512]);  empty_539 = None
        view_1833: "bf16[512, 1024, 14, 14]" = torch.ops.aten.reshape.default(view_1832, [512, 1024, 14, 14]);  view_1832 = None
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(view_1831, view_1833, expand_26, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1831 = view_1833 = expand_26 = None
        getitem_836: "bf16[256, 1024, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
        convert_element_type_127: "f32[256, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_836, torch.float32);  getitem_836 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_231: "bf16[512, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_380, getitem_832);  mul_380 = getitem_832 = None
        
        # No stacktrace found for following nodes
        as_strided_default_212: "i8[102760448]" = torch.ops.aten.as_strided.default(full_default_339, [102760448], [1], 0)
        clone_default_106: "i8[102760448]" = torch.ops.aten.clone.default(as_strided_default_212);  as_strided_default_212 = None
        as_strided_default_213: "i8[200704, 512]" = torch.ops.aten.as_strided.default(clone_default_106, [200704, 512], [512, 1], 0);  clone_default_106 = None
        triton_kernel_wrapper_mutation_196 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 687, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_512, 'Y_ptr': as_strided_default_213, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_512 = triton_kernel_wrapper_mutation_196 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1836: "i8[512, 392, 512]" = torch.ops.aten.reshape.default(as_strided_default_213, [512, 392, 512]);  as_strided_default_213 = None
        view_1837: "i8[512, 1024, 14, 14]" = torch.ops.aten.reshape.default(view_1836, [512, 1024, 14, 14]);  view_1836 = None
        mul_383: "bf16[512, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(add_231, view_1837);  add_231 = view_1837 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_542: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_195 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 488, constant_args_idx = 688, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_507, 'S_ptr': getitem_508, 'M_ptr': getitem_509, 'Y_ptr': empty_542, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_507 = getitem_508 = getitem_509 = triton_kernel_wrapper_mutation_195 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_1852: "bf16[512, 1024, 196]" = torch.ops.aten.reshape.default(mul_383, [512, 1024, 196])
        view_1853: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_542, [512, 392, 512]);  empty_542 = None
        view_1854: "bf16[512, 1024, 196]" = torch.ops.aten.reshape.default(view_1853, [512, 1024, 196]);  view_1853 = None
        
        # No stacktrace found for following nodes
        as_strided_default_208: "f32[1024]" = torch.ops.aten.as_strided.default(full_default_152, [1024], [1], 0)
        clone_default_104: "f32[1024]" = torch.ops.aten.clone.default(as_strided_default_208);  as_strided_default_208 = None
        as_strided_default_209: "f32[1024]" = torch.ops.aten.as_strided.default(clone_default_104, [1024], [1], 0);  clone_default_104 = None
        as_strided_default_210: "f32[1024]" = torch.ops.aten.as_strided.default(full_default_152, [1024], [1], 0)
        clone_default_105: "f32[1024]" = torch.ops.aten.clone.default(as_strided_default_210);  as_strided_default_210 = None
        as_strided_default_211: "f32[1024]" = torch.ops.aten.as_strided.default(clone_default_105, [1024], [1], 0);  clone_default_105 = None
        triton_kernel_wrapper_mutation_194 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 489, constant_args_idx = 689, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1854, 'DY': view_1852, 'DBETA': as_strided_default_209, 'DGAMMA': as_strided_default_211, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_194 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_543: "bf16[512, 1024, 196]" = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_170: "bf16[512, 1024, 196]" = torch.ops.aten.permute.default(empty_543, [0, 1, 2]);  empty_543 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_193 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 490, constant_args_idx = 690, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1854, 'DY': view_1852, 'INVSTD': rsqrt_39, 'GAMMA': primals_238, 'DBETA': as_strided_default_209, 'DGAMMA': as_strided_default_211, 'DX': permute_170, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024});  view_1854 = view_1852 = rsqrt_39 = primals_238 = triton_kernel_wrapper_mutation_193 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_544: "bf16[50176, 512]" = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_192 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 491, constant_args_idx = 691, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_500, 'S_ptr': getitem_501, 'M_ptr': getitem_502, 'Y_ptr': empty_544, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_500 = getitem_501 = getitem_502 = triton_kernel_wrapper_mutation_192 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_1870: "bf16[512, 1024, 14, 14]" = torch.ops.aten.reshape.default(permute_170, [512, 1024, 14, 14]);  permute_170 = None
        empty_545: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_27: "bf16[512, 256, 14, 14]" = torch.ops.aten.expand.default(empty_545, [512, 256, 14, 14]);  empty_545 = None
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(view_1870, expand_27, convert_element_type_40, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_27 = convert_element_type_40 = None
        getitem_844: "bf16[512, 256, 14, 14]" = convolution_backward_26[0];  convolution_backward_26 = None
        empty_546: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_28: "bf16[1024, 256, 1, 1]" = torch.ops.aten.expand.default(empty_546, [1024, 256, 1, 1]);  empty_546 = None
        view_1871: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(empty_544, [512, 98, 512]);  empty_544 = None
        view_1872: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(view_1871, [512, 256, 14, 14]);  view_1871 = None
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(view_1870, view_1872, expand_28, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1870 = view_1872 = expand_28 = None
        getitem_848: "bf16[1024, 256, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
        convert_element_type_132: "f32[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_848, torch.float32);  getitem_848 = None
        
        # No stacktrace found for following nodes
        as_strided_default_206: "i8[25690112]" = torch.ops.aten.as_strided.default(full_default_342, [25690112], [1], 0)
        clone_default_103: "i8[25690112]" = torch.ops.aten.clone.default(as_strided_default_206);  as_strided_default_206 = None
        as_strided_default_207: "i8[50176, 512]" = torch.ops.aten.as_strided.default(clone_default_103, [50176, 512], [512, 1], 0);  clone_default_103 = None
        triton_kernel_wrapper_mutation_191 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 692, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_499, 'Y_ptr': as_strided_default_207, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_499 = triton_kernel_wrapper_mutation_191 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1875: "i8[512, 98, 512]" = torch.ops.aten.reshape.default(as_strided_default_207, [512, 98, 512]);  as_strided_default_207 = None
        view_1876: "i8[512, 256, 14, 14]" = torch.ops.aten.reshape.default(view_1875, [512, 256, 14, 14]);  view_1875 = None
        mul_384: "bf16[512, 256, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_844, view_1876);  getitem_844 = view_1876 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_547: "bf16[50176, 512]" = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_190 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 492, constant_args_idx = 693, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_494, 'S_ptr': getitem_495, 'M_ptr': getitem_496, 'Y_ptr': empty_547, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_494 = getitem_495 = getitem_496 = triton_kernel_wrapper_mutation_190 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_1891: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(mul_384, [512, 256, 196]);  mul_384 = None
        view_1892: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(empty_547, [512, 98, 512]);  empty_547 = None
        view_1893: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(view_1892, [512, 256, 196]);  view_1892 = None
        
        # No stacktrace found for following nodes
        as_strided_default_202: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_101: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_202);  as_strided_default_202 = None
        as_strided_default_203: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_101, [256], [1], 0);  clone_default_101 = None
        as_strided_default_204: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_102: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_204);  as_strided_default_204 = None
        as_strided_default_205: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_102, [256], [1], 0);  clone_default_102 = None
        triton_kernel_wrapper_mutation_189 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 493, constant_args_idx = 694, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1893, 'DY': view_1891, 'DBETA': as_strided_default_203, 'DGAMMA': as_strided_default_205, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_189 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_548: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_171: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_548, [0, 1, 2]);  empty_548 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_188 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 494, constant_args_idx = 695, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1893, 'DY': view_1891, 'INVSTD': rsqrt_38, 'GAMMA': primals_232, 'DBETA': as_strided_default_203, 'DGAMMA': as_strided_default_205, 'DX': permute_171, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  view_1893 = view_1891 = rsqrt_38 = primals_232 = triton_kernel_wrapper_mutation_188 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_549: "bf16[50176, 512]" = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_187 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 495, constant_args_idx = 696, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_487, 'S_ptr': getitem_488, 'M_ptr': getitem_489, 'Y_ptr': empty_549, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_487 = getitem_488 = getitem_489 = triton_kernel_wrapper_mutation_187 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_1909: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(permute_171, [512, 256, 14, 14]);  permute_171 = None
        empty_550: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_29: "bf16[512, 256, 14, 14]" = torch.ops.aten.expand.default(empty_550, [512, 256, 14, 14]);  empty_550 = None
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(view_1909, expand_29, convert_element_type_39, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_29 = convert_element_type_39 = None
        getitem_856: "bf16[512, 256, 14, 14]" = convolution_backward_28[0];  convolution_backward_28 = None
        empty_551: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_30: "bf16[256, 256, 3, 3]" = torch.ops.aten.expand.default(empty_551, [256, 256, 3, 3]);  empty_551 = None
        view_1910: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(empty_549, [512, 98, 512]);  empty_549 = None
        view_1911: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(view_1910, [512, 256, 14, 14]);  view_1910 = None
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(view_1909, view_1911, expand_30, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_1909 = view_1911 = expand_30 = None
        getitem_860: "bf16[256, 256, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None
        convert_element_type_137: "f32[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_860, torch.float32);  getitem_860 = None
        
        # No stacktrace found for following nodes
        as_strided_default_200: "i8[25690112]" = torch.ops.aten.as_strided.default(full_default_342, [25690112], [1], 0)
        clone_default_100: "i8[25690112]" = torch.ops.aten.clone.default(as_strided_default_200);  as_strided_default_200 = None
        as_strided_default_201: "i8[50176, 512]" = torch.ops.aten.as_strided.default(clone_default_100, [50176, 512], [512, 1], 0);  clone_default_100 = None
        triton_kernel_wrapper_mutation_186 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 697, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_486, 'Y_ptr': as_strided_default_201, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_486 = triton_kernel_wrapper_mutation_186 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1914: "i8[512, 98, 512]" = torch.ops.aten.reshape.default(as_strided_default_201, [512, 98, 512]);  as_strided_default_201 = None
        view_1915: "i8[512, 256, 14, 14]" = torch.ops.aten.reshape.default(view_1914, [512, 256, 14, 14]);  view_1914 = None
        mul_385: "bf16[512, 256, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_856, view_1915);  getitem_856 = view_1915 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_552: "bf16[50176, 512]" = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_185 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 496, constant_args_idx = 698, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_481, 'S_ptr': getitem_482, 'M_ptr': getitem_483, 'Y_ptr': empty_552, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_481 = getitem_482 = getitem_483 = triton_kernel_wrapper_mutation_185 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_1930: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(mul_385, [512, 256, 196]);  mul_385 = None
        view_1931: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(empty_552, [512, 98, 512]);  empty_552 = None
        view_1932: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(view_1931, [512, 256, 196]);  view_1931 = None
        
        # No stacktrace found for following nodes
        as_strided_default_196: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_98: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_196);  as_strided_default_196 = None
        as_strided_default_197: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_98, [256], [1], 0);  clone_default_98 = None
        as_strided_default_198: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_99: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_198);  as_strided_default_198 = None
        as_strided_default_199: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_99, [256], [1], 0);  clone_default_99 = None
        triton_kernel_wrapper_mutation_184 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 497, constant_args_idx = 699, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1932, 'DY': view_1930, 'DBETA': as_strided_default_197, 'DGAMMA': as_strided_default_199, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_184 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_553: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_172: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_553, [0, 1, 2]);  empty_553 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_183 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 498, constant_args_idx = 700, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1932, 'DY': view_1930, 'INVSTD': rsqrt_37, 'GAMMA': primals_226, 'DBETA': as_strided_default_197, 'DGAMMA': as_strided_default_199, 'DX': permute_172, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  view_1932 = view_1930 = rsqrt_37 = primals_226 = triton_kernel_wrapper_mutation_183 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_554: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_182 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 499, constant_args_idx = 701, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_474, 'S_ptr': getitem_475, 'M_ptr': getitem_476, 'Y_ptr': empty_554, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_474 = getitem_475 = getitem_476 = triton_kernel_wrapper_mutation_182 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_1948: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(permute_172, [512, 256, 14, 14]);  permute_172 = None
        empty_555: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_31: "bf16[512, 1024, 14, 14]" = torch.ops.aten.expand.default(empty_555, [512, 1024, 14, 14]);  empty_555 = None
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(view_1948, expand_31, convert_element_type_38, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_31 = convert_element_type_38 = None
        getitem_868: "bf16[512, 1024, 14, 14]" = convolution_backward_30[0];  convolution_backward_30 = None
        empty_556: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_32: "bf16[256, 1024, 1, 1]" = torch.ops.aten.expand.default(empty_556, [256, 1024, 1, 1]);  empty_556 = None
        view_1949: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_554, [512, 392, 512]);  empty_554 = None
        view_1950: "bf16[512, 1024, 14, 14]" = torch.ops.aten.reshape.default(view_1949, [512, 1024, 14, 14]);  view_1949 = None
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(view_1948, view_1950, expand_32, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1948 = view_1950 = expand_32 = None
        getitem_872: "bf16[256, 1024, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
        convert_element_type_142: "f32[256, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_872, torch.float32);  getitem_872 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_232: "bf16[512, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_383, getitem_868);  mul_383 = getitem_868 = None
        
        # No stacktrace found for following nodes
        as_strided_default_194: "i8[102760448]" = torch.ops.aten.as_strided.default(full_default_339, [102760448], [1], 0)
        clone_default_97: "i8[102760448]" = torch.ops.aten.clone.default(as_strided_default_194);  as_strided_default_194 = None
        as_strided_default_195: "i8[200704, 512]" = torch.ops.aten.as_strided.default(clone_default_97, [200704, 512], [512, 1], 0);  clone_default_97 = None
        triton_kernel_wrapper_mutation_181 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 702, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_473, 'Y_ptr': as_strided_default_195, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_473 = triton_kernel_wrapper_mutation_181 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1953: "i8[512, 392, 512]" = torch.ops.aten.reshape.default(as_strided_default_195, [512, 392, 512]);  as_strided_default_195 = None
        view_1954: "i8[512, 1024, 14, 14]" = torch.ops.aten.reshape.default(view_1953, [512, 1024, 14, 14]);  view_1953 = None
        mul_386: "bf16[512, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(add_232, view_1954);  add_232 = view_1954 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_557: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_180 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 500, constant_args_idx = 703, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_468, 'S_ptr': getitem_469, 'M_ptr': getitem_470, 'Y_ptr': empty_557, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_468 = getitem_469 = getitem_470 = triton_kernel_wrapper_mutation_180 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_1969: "bf16[512, 1024, 196]" = torch.ops.aten.reshape.default(mul_386, [512, 1024, 196])
        view_1970: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_557, [512, 392, 512]);  empty_557 = None
        view_1971: "bf16[512, 1024, 196]" = torch.ops.aten.reshape.default(view_1970, [512, 1024, 196]);  view_1970 = None
        
        # No stacktrace found for following nodes
        as_strided_default_190: "f32[1024]" = torch.ops.aten.as_strided.default(full_default_152, [1024], [1], 0)
        clone_default_95: "f32[1024]" = torch.ops.aten.clone.default(as_strided_default_190);  as_strided_default_190 = None
        as_strided_default_191: "f32[1024]" = torch.ops.aten.as_strided.default(clone_default_95, [1024], [1], 0);  clone_default_95 = None
        as_strided_default_192: "f32[1024]" = torch.ops.aten.as_strided.default(full_default_152, [1024], [1], 0)
        clone_default_96: "f32[1024]" = torch.ops.aten.clone.default(as_strided_default_192);  as_strided_default_192 = None
        as_strided_default_193: "f32[1024]" = torch.ops.aten.as_strided.default(clone_default_96, [1024], [1], 0);  clone_default_96 = None
        triton_kernel_wrapper_mutation_179 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 501, constant_args_idx = 704, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1971, 'DY': view_1969, 'DBETA': as_strided_default_191, 'DGAMMA': as_strided_default_193, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_179 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_558: "bf16[512, 1024, 196]" = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_173: "bf16[512, 1024, 196]" = torch.ops.aten.permute.default(empty_558, [0, 1, 2]);  empty_558 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_178 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 502, constant_args_idx = 705, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1971, 'DY': view_1969, 'INVSTD': rsqrt_36, 'GAMMA': primals_220, 'DBETA': as_strided_default_191, 'DGAMMA': as_strided_default_193, 'DX': permute_173, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024});  view_1971 = view_1969 = rsqrt_36 = primals_220 = triton_kernel_wrapper_mutation_178 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_559: "bf16[50176, 512]" = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_177 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 503, constant_args_idx = 706, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_461, 'S_ptr': getitem_462, 'M_ptr': getitem_463, 'Y_ptr': empty_559, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_461 = getitem_462 = getitem_463 = triton_kernel_wrapper_mutation_177 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_1987: "bf16[512, 1024, 14, 14]" = torch.ops.aten.reshape.default(permute_173, [512, 1024, 14, 14]);  permute_173 = None
        empty_560: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_33: "bf16[512, 256, 14, 14]" = torch.ops.aten.expand.default(empty_560, [512, 256, 14, 14]);  empty_560 = None
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(view_1987, expand_33, convert_element_type_37, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_33 = convert_element_type_37 = None
        getitem_880: "bf16[512, 256, 14, 14]" = convolution_backward_32[0];  convolution_backward_32 = None
        empty_561: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_34: "bf16[1024, 256, 1, 1]" = torch.ops.aten.expand.default(empty_561, [1024, 256, 1, 1]);  empty_561 = None
        view_1988: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(empty_559, [512, 98, 512]);  empty_559 = None
        view_1989: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(view_1988, [512, 256, 14, 14]);  view_1988 = None
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(view_1987, view_1989, expand_34, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1987 = view_1989 = expand_34 = None
        getitem_884: "bf16[1024, 256, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
        convert_element_type_147: "f32[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_884, torch.float32);  getitem_884 = None
        
        # No stacktrace found for following nodes
        as_strided_default_188: "i8[25690112]" = torch.ops.aten.as_strided.default(full_default_342, [25690112], [1], 0)
        clone_default_94: "i8[25690112]" = torch.ops.aten.clone.default(as_strided_default_188);  as_strided_default_188 = None
        as_strided_default_189: "i8[50176, 512]" = torch.ops.aten.as_strided.default(clone_default_94, [50176, 512], [512, 1], 0);  clone_default_94 = None
        triton_kernel_wrapper_mutation_176 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 707, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_460, 'Y_ptr': as_strided_default_189, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_460 = triton_kernel_wrapper_mutation_176 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_1992: "i8[512, 98, 512]" = torch.ops.aten.reshape.default(as_strided_default_189, [512, 98, 512]);  as_strided_default_189 = None
        view_1993: "i8[512, 256, 14, 14]" = torch.ops.aten.reshape.default(view_1992, [512, 256, 14, 14]);  view_1992 = None
        mul_387: "bf16[512, 256, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_880, view_1993);  getitem_880 = view_1993 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_562: "bf16[50176, 512]" = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_175 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 504, constant_args_idx = 708, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_455, 'S_ptr': getitem_456, 'M_ptr': getitem_457, 'Y_ptr': empty_562, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_455 = getitem_456 = getitem_457 = triton_kernel_wrapper_mutation_175 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_2008: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(mul_387, [512, 256, 196]);  mul_387 = None
        view_2009: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(empty_562, [512, 98, 512]);  empty_562 = None
        view_2010: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(view_2009, [512, 256, 196]);  view_2009 = None
        
        # No stacktrace found for following nodes
        as_strided_default_184: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_92: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_184);  as_strided_default_184 = None
        as_strided_default_185: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_92, [256], [1], 0);  clone_default_92 = None
        as_strided_default_186: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_93: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_186);  as_strided_default_186 = None
        as_strided_default_187: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_93, [256], [1], 0);  clone_default_93 = None
        triton_kernel_wrapper_mutation_174 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 505, constant_args_idx = 709, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2010, 'DY': view_2008, 'DBETA': as_strided_default_185, 'DGAMMA': as_strided_default_187, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_174 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_563: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_174: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_563, [0, 1, 2]);  empty_563 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_173 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 506, constant_args_idx = 710, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2010, 'DY': view_2008, 'INVSTD': rsqrt_35, 'GAMMA': primals_214, 'DBETA': as_strided_default_185, 'DGAMMA': as_strided_default_187, 'DX': permute_174, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  view_2010 = view_2008 = rsqrt_35 = primals_214 = triton_kernel_wrapper_mutation_173 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_564: "bf16[50176, 512]" = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_172 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 507, constant_args_idx = 711, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_448, 'S_ptr': getitem_449, 'M_ptr': getitem_450, 'Y_ptr': empty_564, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_448 = getitem_449 = getitem_450 = triton_kernel_wrapper_mutation_172 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_2026: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(permute_174, [512, 256, 14, 14]);  permute_174 = None
        empty_565: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_35: "bf16[512, 256, 14, 14]" = torch.ops.aten.expand.default(empty_565, [512, 256, 14, 14]);  empty_565 = None
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(view_2026, expand_35, convert_element_type_36, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_35 = convert_element_type_36 = None
        getitem_892: "bf16[512, 256, 14, 14]" = convolution_backward_34[0];  convolution_backward_34 = None
        empty_566: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_36: "bf16[256, 256, 3, 3]" = torch.ops.aten.expand.default(empty_566, [256, 256, 3, 3]);  empty_566 = None
        view_2027: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(empty_564, [512, 98, 512]);  empty_564 = None
        view_2028: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(view_2027, [512, 256, 14, 14]);  view_2027 = None
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(view_2026, view_2028, expand_36, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_2026 = view_2028 = expand_36 = None
        getitem_896: "bf16[256, 256, 3, 3]" = convolution_backward_35[1];  convolution_backward_35 = None
        convert_element_type_152: "f32[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_896, torch.float32);  getitem_896 = None
        
        # No stacktrace found for following nodes
        as_strided_default_182: "i8[25690112]" = torch.ops.aten.as_strided.default(full_default_342, [25690112], [1], 0)
        clone_default_91: "i8[25690112]" = torch.ops.aten.clone.default(as_strided_default_182);  as_strided_default_182 = None
        as_strided_default_183: "i8[50176, 512]" = torch.ops.aten.as_strided.default(clone_default_91, [50176, 512], [512, 1], 0);  clone_default_91 = None
        triton_kernel_wrapper_mutation_171 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 712, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_447, 'Y_ptr': as_strided_default_183, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_447 = triton_kernel_wrapper_mutation_171 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_2031: "i8[512, 98, 512]" = torch.ops.aten.reshape.default(as_strided_default_183, [512, 98, 512]);  as_strided_default_183 = None
        view_2032: "i8[512, 256, 14, 14]" = torch.ops.aten.reshape.default(view_2031, [512, 256, 14, 14]);  view_2031 = None
        mul_388: "bf16[512, 256, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_892, view_2032);  getitem_892 = view_2032 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_567: "bf16[50176, 512]" = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_170 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 508, constant_args_idx = 713, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_442, 'S_ptr': getitem_443, 'M_ptr': getitem_444, 'Y_ptr': empty_567, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_442 = getitem_443 = getitem_444 = triton_kernel_wrapper_mutation_170 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_2047: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(mul_388, [512, 256, 196]);  mul_388 = None
        view_2048: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(empty_567, [512, 98, 512]);  empty_567 = None
        view_2049: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(view_2048, [512, 256, 196]);  view_2048 = None
        
        # No stacktrace found for following nodes
        as_strided_default_178: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_89: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_178);  as_strided_default_178 = None
        as_strided_default_179: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_89, [256], [1], 0);  clone_default_89 = None
        as_strided_default_180: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_90: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_180);  as_strided_default_180 = None
        as_strided_default_181: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_90, [256], [1], 0);  clone_default_90 = None
        triton_kernel_wrapper_mutation_169 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 509, constant_args_idx = 714, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2049, 'DY': view_2047, 'DBETA': as_strided_default_179, 'DGAMMA': as_strided_default_181, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_169 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_568: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_175: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_568, [0, 1, 2]);  empty_568 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_168 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 510, constant_args_idx = 715, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2049, 'DY': view_2047, 'INVSTD': rsqrt_34, 'GAMMA': primals_208, 'DBETA': as_strided_default_179, 'DGAMMA': as_strided_default_181, 'DX': permute_175, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  view_2049 = view_2047 = rsqrt_34 = primals_208 = triton_kernel_wrapper_mutation_168 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_569: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_167 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 511, constant_args_idx = 716, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_435, 'S_ptr': getitem_436, 'M_ptr': getitem_437, 'Y_ptr': empty_569, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_435 = getitem_436 = getitem_437 = triton_kernel_wrapper_mutation_167 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_2065: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(permute_175, [512, 256, 14, 14]);  permute_175 = None
        empty_570: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_37: "bf16[512, 1024, 14, 14]" = torch.ops.aten.expand.default(empty_570, [512, 1024, 14, 14]);  empty_570 = None
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(view_2065, expand_37, convert_element_type_35, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_37 = convert_element_type_35 = None
        getitem_904: "bf16[512, 1024, 14, 14]" = convolution_backward_36[0];  convolution_backward_36 = None
        empty_571: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_38: "bf16[256, 1024, 1, 1]" = torch.ops.aten.expand.default(empty_571, [256, 1024, 1, 1]);  empty_571 = None
        view_2066: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_569, [512, 392, 512]);  empty_569 = None
        view_2067: "bf16[512, 1024, 14, 14]" = torch.ops.aten.reshape.default(view_2066, [512, 1024, 14, 14]);  view_2066 = None
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(view_2065, view_2067, expand_38, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2065 = view_2067 = expand_38 = None
        getitem_908: "bf16[256, 1024, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
        convert_element_type_157: "f32[256, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_908, torch.float32);  getitem_908 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_233: "bf16[512, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_386, getitem_904);  mul_386 = getitem_904 = None
        
        # No stacktrace found for following nodes
        as_strided_default_176: "i8[102760448]" = torch.ops.aten.as_strided.default(full_default_339, [102760448], [1], 0)
        clone_default_88: "i8[102760448]" = torch.ops.aten.clone.default(as_strided_default_176);  as_strided_default_176 = None
        as_strided_default_177: "i8[200704, 512]" = torch.ops.aten.as_strided.default(clone_default_88, [200704, 512], [512, 1], 0);  clone_default_88 = None
        triton_kernel_wrapper_mutation_166 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 717, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_434, 'Y_ptr': as_strided_default_177, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_434 = triton_kernel_wrapper_mutation_166 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_2070: "i8[512, 392, 512]" = torch.ops.aten.reshape.default(as_strided_default_177, [512, 392, 512]);  as_strided_default_177 = None
        view_2071: "i8[512, 1024, 14, 14]" = torch.ops.aten.reshape.default(view_2070, [512, 1024, 14, 14]);  view_2070 = None
        mul_389: "bf16[512, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(add_233, view_2071);  add_233 = view_2071 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_572: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_165 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 512, constant_args_idx = 718, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_429, 'S_ptr': getitem_430, 'M_ptr': getitem_431, 'Y_ptr': empty_572, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_429 = getitem_430 = getitem_431 = triton_kernel_wrapper_mutation_165 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_2086: "bf16[512, 1024, 196]" = torch.ops.aten.reshape.default(mul_389, [512, 1024, 196])
        view_2087: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_572, [512, 392, 512]);  empty_572 = None
        view_2088: "bf16[512, 1024, 196]" = torch.ops.aten.reshape.default(view_2087, [512, 1024, 196]);  view_2087 = None
        
        # No stacktrace found for following nodes
        as_strided_default_172: "f32[1024]" = torch.ops.aten.as_strided.default(full_default_152, [1024], [1], 0)
        clone_default_86: "f32[1024]" = torch.ops.aten.clone.default(as_strided_default_172);  as_strided_default_172 = None
        as_strided_default_173: "f32[1024]" = torch.ops.aten.as_strided.default(clone_default_86, [1024], [1], 0);  clone_default_86 = None
        as_strided_default_174: "f32[1024]" = torch.ops.aten.as_strided.default(full_default_152, [1024], [1], 0)
        clone_default_87: "f32[1024]" = torch.ops.aten.clone.default(as_strided_default_174);  as_strided_default_174 = None
        as_strided_default_175: "f32[1024]" = torch.ops.aten.as_strided.default(clone_default_87, [1024], [1], 0);  clone_default_87 = None
        triton_kernel_wrapper_mutation_164 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 513, constant_args_idx = 719, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2088, 'DY': view_2086, 'DBETA': as_strided_default_173, 'DGAMMA': as_strided_default_175, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_164 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_573: "bf16[512, 1024, 196]" = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_176: "bf16[512, 1024, 196]" = torch.ops.aten.permute.default(empty_573, [0, 1, 2]);  empty_573 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_163 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 514, constant_args_idx = 720, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2088, 'DY': view_2086, 'INVSTD': rsqrt_33, 'GAMMA': primals_202, 'DBETA': as_strided_default_173, 'DGAMMA': as_strided_default_175, 'DX': permute_176, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024});  view_2088 = view_2086 = rsqrt_33 = primals_202 = triton_kernel_wrapper_mutation_163 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_574: "bf16[50176, 512]" = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_162 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 515, constant_args_idx = 721, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_422, 'S_ptr': getitem_423, 'M_ptr': getitem_424, 'Y_ptr': empty_574, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_422 = getitem_423 = getitem_424 = triton_kernel_wrapper_mutation_162 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_2104: "bf16[512, 1024, 14, 14]" = torch.ops.aten.reshape.default(permute_176, [512, 1024, 14, 14]);  permute_176 = None
        empty_575: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_39: "bf16[512, 256, 14, 14]" = torch.ops.aten.expand.default(empty_575, [512, 256, 14, 14]);  empty_575 = None
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(view_2104, expand_39, convert_element_type_34, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_39 = convert_element_type_34 = None
        getitem_916: "bf16[512, 256, 14, 14]" = convolution_backward_38[0];  convolution_backward_38 = None
        empty_576: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_40: "bf16[1024, 256, 1, 1]" = torch.ops.aten.expand.default(empty_576, [1024, 256, 1, 1]);  empty_576 = None
        view_2105: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(empty_574, [512, 98, 512]);  empty_574 = None
        view_2106: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(view_2105, [512, 256, 14, 14]);  view_2105 = None
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(view_2104, view_2106, expand_40, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2104 = view_2106 = expand_40 = None
        getitem_920: "bf16[1024, 256, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None
        convert_element_type_162: "f32[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_920, torch.float32);  getitem_920 = None
        
        # No stacktrace found for following nodes
        as_strided_default_170: "i8[25690112]" = torch.ops.aten.as_strided.default(full_default_342, [25690112], [1], 0)
        clone_default_85: "i8[25690112]" = torch.ops.aten.clone.default(as_strided_default_170);  as_strided_default_170 = None
        as_strided_default_171: "i8[50176, 512]" = torch.ops.aten.as_strided.default(clone_default_85, [50176, 512], [512, 1], 0);  clone_default_85 = None
        triton_kernel_wrapper_mutation_161 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 722, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_421, 'Y_ptr': as_strided_default_171, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_421 = triton_kernel_wrapper_mutation_161 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_2109: "i8[512, 98, 512]" = torch.ops.aten.reshape.default(as_strided_default_171, [512, 98, 512]);  as_strided_default_171 = None
        view_2110: "i8[512, 256, 14, 14]" = torch.ops.aten.reshape.default(view_2109, [512, 256, 14, 14]);  view_2109 = None
        mul_390: "bf16[512, 256, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_916, view_2110);  getitem_916 = view_2110 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_577: "bf16[50176, 512]" = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_160 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 516, constant_args_idx = 723, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_416, 'S_ptr': getitem_417, 'M_ptr': getitem_418, 'Y_ptr': empty_577, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_416 = getitem_417 = getitem_418 = triton_kernel_wrapper_mutation_160 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_2125: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(mul_390, [512, 256, 196]);  mul_390 = None
        view_2126: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(empty_577, [512, 98, 512]);  empty_577 = None
        view_2127: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(view_2126, [512, 256, 196]);  view_2126 = None
        
        # No stacktrace found for following nodes
        as_strided_default_166: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_83: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_166);  as_strided_default_166 = None
        as_strided_default_167: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_83, [256], [1], 0);  clone_default_83 = None
        as_strided_default_168: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_84: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_168);  as_strided_default_168 = None
        as_strided_default_169: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_84, [256], [1], 0);  clone_default_84 = None
        triton_kernel_wrapper_mutation_159 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 517, constant_args_idx = 724, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2127, 'DY': view_2125, 'DBETA': as_strided_default_167, 'DGAMMA': as_strided_default_169, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_159 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_578: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_177: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_578, [0, 1, 2]);  empty_578 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_158 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 518, constant_args_idx = 725, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2127, 'DY': view_2125, 'INVSTD': rsqrt_32, 'GAMMA': primals_196, 'DBETA': as_strided_default_167, 'DGAMMA': as_strided_default_169, 'DX': permute_177, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  view_2127 = view_2125 = rsqrt_32 = primals_196 = triton_kernel_wrapper_mutation_158 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_579: "bf16[50176, 512]" = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_157 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 519, constant_args_idx = 726, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_409, 'S_ptr': getitem_410, 'M_ptr': getitem_411, 'Y_ptr': empty_579, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_409 = getitem_410 = getitem_411 = triton_kernel_wrapper_mutation_157 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_2143: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(permute_177, [512, 256, 14, 14]);  permute_177 = None
        empty_580: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_41: "bf16[512, 256, 14, 14]" = torch.ops.aten.expand.default(empty_580, [512, 256, 14, 14]);  empty_580 = None
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(view_2143, expand_41, convert_element_type_33, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_41 = convert_element_type_33 = None
        getitem_928: "bf16[512, 256, 14, 14]" = convolution_backward_40[0];  convolution_backward_40 = None
        empty_581: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_42: "bf16[256, 256, 3, 3]" = torch.ops.aten.expand.default(empty_581, [256, 256, 3, 3]);  empty_581 = None
        view_2144: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(empty_579, [512, 98, 512]);  empty_579 = None
        view_2145: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(view_2144, [512, 256, 14, 14]);  view_2144 = None
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(view_2143, view_2145, expand_42, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_2143 = view_2145 = expand_42 = None
        getitem_932: "bf16[256, 256, 3, 3]" = convolution_backward_41[1];  convolution_backward_41 = None
        convert_element_type_167: "f32[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_932, torch.float32);  getitem_932 = None
        
        # No stacktrace found for following nodes
        as_strided_default_164: "i8[25690112]" = torch.ops.aten.as_strided.default(full_default_342, [25690112], [1], 0)
        clone_default_82: "i8[25690112]" = torch.ops.aten.clone.default(as_strided_default_164);  as_strided_default_164 = None
        as_strided_default_165: "i8[50176, 512]" = torch.ops.aten.as_strided.default(clone_default_82, [50176, 512], [512, 1], 0);  clone_default_82 = None
        triton_kernel_wrapper_mutation_156 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 727, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_408, 'Y_ptr': as_strided_default_165, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_408 = triton_kernel_wrapper_mutation_156 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_2148: "i8[512, 98, 512]" = torch.ops.aten.reshape.default(as_strided_default_165, [512, 98, 512]);  as_strided_default_165 = None
        view_2149: "i8[512, 256, 14, 14]" = torch.ops.aten.reshape.default(view_2148, [512, 256, 14, 14]);  view_2148 = None
        mul_391: "bf16[512, 256, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_928, view_2149);  getitem_928 = view_2149 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_582: "bf16[50176, 512]" = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_155 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 520, constant_args_idx = 728, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_403, 'S_ptr': getitem_404, 'M_ptr': getitem_405, 'Y_ptr': empty_582, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_403 = getitem_404 = getitem_405 = triton_kernel_wrapper_mutation_155 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_2164: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(mul_391, [512, 256, 196]);  mul_391 = None
        view_2165: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(empty_582, [512, 98, 512]);  empty_582 = None
        view_2166: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(view_2165, [512, 256, 196]);  view_2165 = None
        
        # No stacktrace found for following nodes
        as_strided_default_160: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_80: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_160);  as_strided_default_160 = None
        as_strided_default_161: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_80, [256], [1], 0);  clone_default_80 = None
        as_strided_default_162: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_81: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_162);  as_strided_default_162 = None
        as_strided_default_163: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_81, [256], [1], 0);  clone_default_81 = None
        triton_kernel_wrapper_mutation_154 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 521, constant_args_idx = 729, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2166, 'DY': view_2164, 'DBETA': as_strided_default_161, 'DGAMMA': as_strided_default_163, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_154 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_583: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_178: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_583, [0, 1, 2]);  empty_583 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_153 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 522, constant_args_idx = 730, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2166, 'DY': view_2164, 'INVSTD': rsqrt_31, 'GAMMA': primals_190, 'DBETA': as_strided_default_161, 'DGAMMA': as_strided_default_163, 'DX': permute_178, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  view_2166 = view_2164 = rsqrt_31 = primals_190 = triton_kernel_wrapper_mutation_153 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_584: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_152 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 523, constant_args_idx = 731, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_396, 'S_ptr': getitem_397, 'M_ptr': getitem_398, 'Y_ptr': empty_584, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_396 = getitem_397 = getitem_398 = triton_kernel_wrapper_mutation_152 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_2182: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(permute_178, [512, 256, 14, 14]);  permute_178 = None
        empty_585: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_43: "bf16[512, 1024, 14, 14]" = torch.ops.aten.expand.default(empty_585, [512, 1024, 14, 14]);  empty_585 = None
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(view_2182, expand_43, convert_element_type_32, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_43 = convert_element_type_32 = None
        getitem_940: "bf16[512, 1024, 14, 14]" = convolution_backward_42[0];  convolution_backward_42 = None
        empty_586: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_44: "bf16[256, 1024, 1, 1]" = torch.ops.aten.expand.default(empty_586, [256, 1024, 1, 1]);  empty_586 = None
        view_2183: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_584, [512, 392, 512]);  empty_584 = None
        view_2184: "bf16[512, 1024, 14, 14]" = torch.ops.aten.reshape.default(view_2183, [512, 1024, 14, 14]);  view_2183 = None
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(view_2182, view_2184, expand_44, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2182 = view_2184 = expand_44 = None
        getitem_944: "bf16[256, 1024, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
        convert_element_type_172: "f32[256, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_944, torch.float32);  getitem_944 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_234: "bf16[512, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_389, getitem_940);  mul_389 = getitem_940 = None
        
        # No stacktrace found for following nodes
        as_strided_default_158: "i8[102760448]" = torch.ops.aten.as_strided.default(full_default_339, [102760448], [1], 0)
        clone_default_79: "i8[102760448]" = torch.ops.aten.clone.default(as_strided_default_158);  as_strided_default_158 = None
        as_strided_default_159: "i8[200704, 512]" = torch.ops.aten.as_strided.default(clone_default_79, [200704, 512], [512, 1], 0);  clone_default_79 = None
        triton_kernel_wrapper_mutation_151 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 732, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_395, 'Y_ptr': as_strided_default_159, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_395 = triton_kernel_wrapper_mutation_151 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_2187: "i8[512, 392, 512]" = torch.ops.aten.reshape.default(as_strided_default_159, [512, 392, 512]);  as_strided_default_159 = None
        view_2188: "i8[512, 1024, 14, 14]" = torch.ops.aten.reshape.default(view_2187, [512, 1024, 14, 14]);  view_2187 = None
        mul_392: "bf16[512, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(add_234, view_2188);  add_234 = view_2188 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_587: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_150 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 524, constant_args_idx = 733, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_390, 'S_ptr': getitem_391, 'M_ptr': getitem_392, 'Y_ptr': empty_587, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_390 = getitem_391 = getitem_392 = triton_kernel_wrapper_mutation_150 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_2203: "bf16[512, 1024, 196]" = torch.ops.aten.reshape.default(mul_392, [512, 1024, 196])
        view_2204: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_587, [512, 392, 512]);  empty_587 = None
        view_2205: "bf16[512, 1024, 196]" = torch.ops.aten.reshape.default(view_2204, [512, 1024, 196]);  view_2204 = None
        
        # No stacktrace found for following nodes
        as_strided_default_154: "f32[1024]" = torch.ops.aten.as_strided.default(full_default_152, [1024], [1], 0)
        clone_default_77: "f32[1024]" = torch.ops.aten.clone.default(as_strided_default_154);  as_strided_default_154 = None
        as_strided_default_155: "f32[1024]" = torch.ops.aten.as_strided.default(clone_default_77, [1024], [1], 0);  clone_default_77 = None
        as_strided_default_156: "f32[1024]" = torch.ops.aten.as_strided.default(full_default_152, [1024], [1], 0)
        clone_default_78: "f32[1024]" = torch.ops.aten.clone.default(as_strided_default_156);  as_strided_default_156 = None
        as_strided_default_157: "f32[1024]" = torch.ops.aten.as_strided.default(clone_default_78, [1024], [1], 0);  clone_default_78 = None
        triton_kernel_wrapper_mutation_149 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 525, constant_args_idx = 734, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2205, 'DY': view_2203, 'DBETA': as_strided_default_155, 'DGAMMA': as_strided_default_157, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_149 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_588: "bf16[512, 1024, 196]" = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_179: "bf16[512, 1024, 196]" = torch.ops.aten.permute.default(empty_588, [0, 1, 2]);  empty_588 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_148 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 526, constant_args_idx = 735, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2205, 'DY': view_2203, 'INVSTD': rsqrt_30, 'GAMMA': primals_184, 'DBETA': as_strided_default_155, 'DGAMMA': as_strided_default_157, 'DX': permute_179, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024});  view_2205 = view_2203 = rsqrt_30 = primals_184 = triton_kernel_wrapper_mutation_148 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_589: "bf16[50176, 512]" = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_147 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 527, constant_args_idx = 736, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_383, 'S_ptr': getitem_384, 'M_ptr': getitem_385, 'Y_ptr': empty_589, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_383 = getitem_384 = getitem_385 = triton_kernel_wrapper_mutation_147 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_2221: "bf16[512, 1024, 14, 14]" = torch.ops.aten.reshape.default(permute_179, [512, 1024, 14, 14]);  permute_179 = None
        empty_590: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_45: "bf16[512, 256, 14, 14]" = torch.ops.aten.expand.default(empty_590, [512, 256, 14, 14]);  empty_590 = None
        convolution_backward_44 = torch.ops.aten.convolution_backward.default(view_2221, expand_45, convert_element_type_31, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_45 = convert_element_type_31 = None
        getitem_952: "bf16[512, 256, 14, 14]" = convolution_backward_44[0];  convolution_backward_44 = None
        empty_591: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_46: "bf16[1024, 256, 1, 1]" = torch.ops.aten.expand.default(empty_591, [1024, 256, 1, 1]);  empty_591 = None
        view_2222: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(empty_589, [512, 98, 512]);  empty_589 = None
        view_2223: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(view_2222, [512, 256, 14, 14]);  view_2222 = None
        convolution_backward_45 = torch.ops.aten.convolution_backward.default(view_2221, view_2223, expand_46, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2221 = view_2223 = expand_46 = None
        getitem_956: "bf16[1024, 256, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
        convert_element_type_177: "f32[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_956, torch.float32);  getitem_956 = None
        
        # No stacktrace found for following nodes
        as_strided_default_152: "i8[25690112]" = torch.ops.aten.as_strided.default(full_default_342, [25690112], [1], 0)
        clone_default_76: "i8[25690112]" = torch.ops.aten.clone.default(as_strided_default_152);  as_strided_default_152 = None
        as_strided_default_153: "i8[50176, 512]" = torch.ops.aten.as_strided.default(clone_default_76, [50176, 512], [512, 1], 0);  clone_default_76 = None
        triton_kernel_wrapper_mutation_146 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 737, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_382, 'Y_ptr': as_strided_default_153, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_382 = triton_kernel_wrapper_mutation_146 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_2226: "i8[512, 98, 512]" = torch.ops.aten.reshape.default(as_strided_default_153, [512, 98, 512]);  as_strided_default_153 = None
        view_2227: "i8[512, 256, 14, 14]" = torch.ops.aten.reshape.default(view_2226, [512, 256, 14, 14]);  view_2226 = None
        mul_393: "bf16[512, 256, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_952, view_2227);  getitem_952 = view_2227 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_592: "bf16[50176, 512]" = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_145 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 528, constant_args_idx = 738, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_377, 'S_ptr': getitem_378, 'M_ptr': getitem_379, 'Y_ptr': empty_592, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_377 = getitem_378 = getitem_379 = triton_kernel_wrapper_mutation_145 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_2242: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(mul_393, [512, 256, 196]);  mul_393 = None
        view_2243: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(empty_592, [512, 98, 512]);  empty_592 = None
        view_2244: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(view_2243, [512, 256, 196]);  view_2243 = None
        
        # No stacktrace found for following nodes
        as_strided_default_148: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_74: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_148);  as_strided_default_148 = None
        as_strided_default_149: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_74, [256], [1], 0);  clone_default_74 = None
        as_strided_default_150: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_75: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_150);  as_strided_default_150 = None
        as_strided_default_151: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_75, [256], [1], 0);  clone_default_75 = None
        triton_kernel_wrapper_mutation_144 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 529, constant_args_idx = 739, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2244, 'DY': view_2242, 'DBETA': as_strided_default_149, 'DGAMMA': as_strided_default_151, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_144 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_593: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_180: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_593, [0, 1, 2]);  empty_593 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_143 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 530, constant_args_idx = 740, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2244, 'DY': view_2242, 'INVSTD': rsqrt_29, 'GAMMA': primals_178, 'DBETA': as_strided_default_149, 'DGAMMA': as_strided_default_151, 'DX': permute_180, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  view_2244 = view_2242 = rsqrt_29 = primals_178 = triton_kernel_wrapper_mutation_143 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_594: "bf16[50176, 512]" = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_142 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 531, constant_args_idx = 741, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_370, 'S_ptr': getitem_371, 'M_ptr': getitem_372, 'Y_ptr': empty_594, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_370 = getitem_371 = getitem_372 = triton_kernel_wrapper_mutation_142 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_2260: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(permute_180, [512, 256, 14, 14]);  permute_180 = None
        empty_595: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_47: "bf16[512, 256, 14, 14]" = torch.ops.aten.expand.default(empty_595, [512, 256, 14, 14]);  empty_595 = None
        convolution_backward_46 = torch.ops.aten.convolution_backward.default(view_2260, expand_47, convert_element_type_30, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_47 = convert_element_type_30 = None
        getitem_964: "bf16[512, 256, 14, 14]" = convolution_backward_46[0];  convolution_backward_46 = None
        empty_596: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_48: "bf16[256, 256, 3, 3]" = torch.ops.aten.expand.default(empty_596, [256, 256, 3, 3]);  empty_596 = None
        view_2261: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(empty_594, [512, 98, 512]);  empty_594 = None
        view_2262: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(view_2261, [512, 256, 14, 14]);  view_2261 = None
        convolution_backward_47 = torch.ops.aten.convolution_backward.default(view_2260, view_2262, expand_48, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_2260 = view_2262 = expand_48 = None
        getitem_968: "bf16[256, 256, 3, 3]" = convolution_backward_47[1];  convolution_backward_47 = None
        convert_element_type_182: "f32[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_968, torch.float32);  getitem_968 = None
        
        # No stacktrace found for following nodes
        as_strided_default_146: "i8[25690112]" = torch.ops.aten.as_strided.default(full_default_342, [25690112], [1], 0)
        clone_default_73: "i8[25690112]" = torch.ops.aten.clone.default(as_strided_default_146);  as_strided_default_146 = None
        as_strided_default_147: "i8[50176, 512]" = torch.ops.aten.as_strided.default(clone_default_73, [50176, 512], [512, 1], 0);  clone_default_73 = None
        triton_kernel_wrapper_mutation_141 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 742, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_369, 'Y_ptr': as_strided_default_147, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_369 = triton_kernel_wrapper_mutation_141 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_2265: "i8[512, 98, 512]" = torch.ops.aten.reshape.default(as_strided_default_147, [512, 98, 512]);  as_strided_default_147 = None
        view_2266: "i8[512, 256, 14, 14]" = torch.ops.aten.reshape.default(view_2265, [512, 256, 14, 14]);  view_2265 = None
        mul_394: "bf16[512, 256, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_964, view_2266);  getitem_964 = view_2266 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_597: "bf16[50176, 512]" = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_140 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 532, constant_args_idx = 743, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_364, 'S_ptr': getitem_365, 'M_ptr': getitem_366, 'Y_ptr': empty_597, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_364 = getitem_365 = getitem_366 = triton_kernel_wrapper_mutation_140 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_2281: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(mul_394, [512, 256, 196]);  mul_394 = None
        view_2282: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(empty_597, [512, 98, 512]);  empty_597 = None
        view_2283: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(view_2282, [512, 256, 196]);  view_2282 = None
        
        # No stacktrace found for following nodes
        as_strided_default_142: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_71: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_142);  as_strided_default_142 = None
        as_strided_default_143: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_71, [256], [1], 0);  clone_default_71 = None
        as_strided_default_144: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_72: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_144);  as_strided_default_144 = None
        as_strided_default_145: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_72, [256], [1], 0);  clone_default_72 = None
        triton_kernel_wrapper_mutation_139 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 533, constant_args_idx = 744, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2283, 'DY': view_2281, 'DBETA': as_strided_default_143, 'DGAMMA': as_strided_default_145, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_139 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_598: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_181: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_598, [0, 1, 2]);  empty_598 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_138 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 534, constant_args_idx = 745, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2283, 'DY': view_2281, 'INVSTD': rsqrt_28, 'GAMMA': primals_172, 'DBETA': as_strided_default_143, 'DGAMMA': as_strided_default_145, 'DX': permute_181, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  view_2283 = view_2281 = rsqrt_28 = primals_172 = triton_kernel_wrapper_mutation_138 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_599: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_137 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 535, constant_args_idx = 746, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_357, 'S_ptr': getitem_358, 'M_ptr': getitem_359, 'Y_ptr': empty_599, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_357 = getitem_358 = getitem_359 = triton_kernel_wrapper_mutation_137 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_2299: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(permute_181, [512, 256, 14, 14]);  permute_181 = None
        empty_600: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_49: "bf16[512, 1024, 14, 14]" = torch.ops.aten.expand.default(empty_600, [512, 1024, 14, 14]);  empty_600 = None
        convolution_backward_48 = torch.ops.aten.convolution_backward.default(view_2299, expand_49, convert_element_type_29, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_49 = convert_element_type_29 = None
        getitem_976: "bf16[512, 1024, 14, 14]" = convolution_backward_48[0];  convolution_backward_48 = None
        empty_601: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_50: "bf16[256, 1024, 1, 1]" = torch.ops.aten.expand.default(empty_601, [256, 1024, 1, 1]);  empty_601 = None
        view_2300: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_599, [512, 392, 512]);  empty_599 = None
        view_2301: "bf16[512, 1024, 14, 14]" = torch.ops.aten.reshape.default(view_2300, [512, 1024, 14, 14]);  view_2300 = None
        convolution_backward_49 = torch.ops.aten.convolution_backward.default(view_2299, view_2301, expand_50, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2299 = view_2301 = expand_50 = None
        getitem_980: "bf16[256, 1024, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None
        convert_element_type_187: "f32[256, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_980, torch.float32);  getitem_980 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_235: "bf16[512, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_392, getitem_976);  mul_392 = getitem_976 = None
        
        # No stacktrace found for following nodes
        as_strided_default_140: "i8[102760448]" = torch.ops.aten.as_strided.default(full_default_339, [102760448], [1], 0)
        clone_default_70: "i8[102760448]" = torch.ops.aten.clone.default(as_strided_default_140);  as_strided_default_140 = None
        as_strided_default_141: "i8[200704, 512]" = torch.ops.aten.as_strided.default(clone_default_70, [200704, 512], [512, 1], 0);  clone_default_70 = None
        triton_kernel_wrapper_mutation_136 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 747, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_356, 'Y_ptr': as_strided_default_141, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_356 = triton_kernel_wrapper_mutation_136 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_2304: "i8[512, 392, 512]" = torch.ops.aten.reshape.default(as_strided_default_141, [512, 392, 512]);  as_strided_default_141 = None
        view_2305: "i8[512, 1024, 14, 14]" = torch.ops.aten.reshape.default(view_2304, [512, 1024, 14, 14]);  view_2304 = None
        mul_395: "bf16[512, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(add_235, view_2305);  add_235 = view_2305 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_602: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_135 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 536, constant_args_idx = 748, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_351, 'S_ptr': getitem_352, 'M_ptr': getitem_353, 'Y_ptr': empty_602, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_351 = getitem_352 = getitem_353 = triton_kernel_wrapper_mutation_135 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_2320: "bf16[512, 1024, 196]" = torch.ops.aten.reshape.default(mul_395, [512, 1024, 196]);  mul_395 = None
        view_2321: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_602, [512, 392, 512]);  empty_602 = None
        view_2322: "bf16[512, 1024, 196]" = torch.ops.aten.reshape.default(view_2321, [512, 1024, 196]);  view_2321 = None
        
        # No stacktrace found for following nodes
        as_strided_default_136: "f32[1024]" = torch.ops.aten.as_strided.default(full_default_152, [1024], [1], 0)
        clone_default_68: "f32[1024]" = torch.ops.aten.clone.default(as_strided_default_136);  as_strided_default_136 = None
        as_strided_default_137: "f32[1024]" = torch.ops.aten.as_strided.default(clone_default_68, [1024], [1], 0);  clone_default_68 = None
        as_strided_default_138: "f32[1024]" = torch.ops.aten.as_strided.default(full_default_152, [1024], [1], 0)
        clone_default_69: "f32[1024]" = torch.ops.aten.clone.default(as_strided_default_138);  as_strided_default_138 = None
        as_strided_default_139: "f32[1024]" = torch.ops.aten.as_strided.default(clone_default_69, [1024], [1], 0);  clone_default_69 = None
        triton_kernel_wrapper_mutation_134 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 537, constant_args_idx = 749, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2322, 'DY': view_2320, 'DBETA': as_strided_default_137, 'DGAMMA': as_strided_default_139, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_134 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_603: "bf16[512, 1024, 196]" = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_182: "bf16[512, 1024, 196]" = torch.ops.aten.permute.default(empty_603, [0, 1, 2]);  empty_603 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_133 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 538, constant_args_idx = 750, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2322, 'DY': view_2320, 'INVSTD': rsqrt_27, 'GAMMA': primals_166, 'DBETA': as_strided_default_137, 'DGAMMA': as_strided_default_139, 'DX': permute_182, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024});  view_2322 = rsqrt_27 = primals_166 = triton_kernel_wrapper_mutation_133 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_604: "bf16[401408, 512]" = torch.ops.aten.empty.memory_format([401408, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_132 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 539, constant_args_idx = 751, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_344, 'S_ptr': getitem_345, 'M_ptr': getitem_346, 'Y_ptr': empty_604, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_344 = getitem_345 = getitem_346 = triton_kernel_wrapper_mutation_132 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_2338: "bf16[512, 1024, 14, 14]" = torch.ops.aten.reshape.default(permute_182, [512, 1024, 14, 14]);  permute_182 = None
        empty_605: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_51: "bf16[512, 512, 28, 28]" = torch.ops.aten.expand.default(empty_605, [512, 512, 28, 28]);  empty_605 = None
        convolution_backward_50 = torch.ops.aten.convolution_backward.default(view_2338, expand_51, convert_element_type_28, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_51 = convert_element_type_28 = None
        getitem_988: "bf16[512, 512, 28, 28]" = convolution_backward_50[0];  convolution_backward_50 = None
        empty_606: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_52: "bf16[1024, 512, 1, 1]" = torch.ops.aten.expand.default(empty_606, [1024, 512, 1, 1]);  empty_606 = None
        view_2339: "bf16[512, 784, 512]" = torch.ops.aten.reshape.default(empty_604, [512, 784, 512]);  empty_604 = None
        view_2340: "bf16[512, 512, 28, 28]" = torch.ops.aten.reshape.default(view_2339, [512, 512, 28, 28]);  view_2339 = None
        convolution_backward_51 = torch.ops.aten.convolution_backward.default(view_2338, view_2340, expand_52, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2338 = view_2340 = expand_52 = None
        getitem_992: "bf16[1024, 512, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
        convert_element_type_192: "f32[1024, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_992, torch.float32);  getitem_992 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_607: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_131 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 540, constant_args_idx = 752, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_341, 'S_ptr': getitem_342, 'M_ptr': getitem_343, 'Y_ptr': empty_607, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_341 = getitem_342 = getitem_343 = triton_kernel_wrapper_mutation_131 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_2356: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_607, [512, 392, 512]);  empty_607 = None
        view_2357: "bf16[512, 1024, 196]" = torch.ops.aten.reshape.default(view_2356, [512, 1024, 196]);  view_2356 = None
        
        # No stacktrace found for following nodes
        as_strided_default_134: "f32[1024]" = torch.ops.aten.as_strided.default(full_default_152, [1024], [1], 0)
        clone_default_67: "f32[1024]" = torch.ops.aten.clone.default(as_strided_default_134);  as_strided_default_134 = None
        as_strided_default_135: "f32[1024]" = torch.ops.aten.as_strided.default(clone_default_67, [1024], [1], 0);  clone_default_67 = None
        triton_kernel_wrapper_mutation_130 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 541, constant_args_idx = 753, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2357, 'DY': view_2320, 'DBETA': full_default_152, 'DGAMMA': as_strided_default_135, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_130 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_608: "bf16[512, 1024, 196]" = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_183: "bf16[512, 1024, 196]" = torch.ops.aten.permute.default(empty_608, [0, 1, 2]);  empty_608 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_129 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 542, constant_args_idx = 754, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2357, 'DY': view_2320, 'INVSTD': rsqrt_26, 'GAMMA': primals_160, 'DBETA': full_default_152, 'DGAMMA': as_strided_default_135, 'DX': permute_183, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024});  view_2357 = view_2320 = rsqrt_26 = primals_160 = triton_kernel_wrapper_mutation_129 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_609: "bf16[50176, 512]" = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_128 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 543, constant_args_idx = 755, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_334, 'S_ptr': getitem_335, 'M_ptr': getitem_336, 'Y_ptr': empty_609, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_334 = getitem_335 = getitem_336 = triton_kernel_wrapper_mutation_128 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_2373: "bf16[512, 1024, 14, 14]" = torch.ops.aten.reshape.default(permute_183, [512, 1024, 14, 14]);  permute_183 = None
        empty_610: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_53: "bf16[512, 256, 14, 14]" = torch.ops.aten.expand.default(empty_610, [512, 256, 14, 14]);  empty_610 = None
        convolution_backward_52 = torch.ops.aten.convolution_backward.default(view_2373, expand_53, convert_element_type_27, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_53 = convert_element_type_27 = None
        getitem_999: "bf16[512, 256, 14, 14]" = convolution_backward_52[0];  convolution_backward_52 = None
        empty_611: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_54: "bf16[1024, 256, 1, 1]" = torch.ops.aten.expand.default(empty_611, [1024, 256, 1, 1]);  empty_611 = None
        view_2374: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(empty_609, [512, 98, 512]);  empty_609 = None
        view_2375: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(view_2374, [512, 256, 14, 14]);  view_2374 = None
        convolution_backward_53 = torch.ops.aten.convolution_backward.default(view_2373, view_2375, expand_54, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2373 = view_2375 = expand_54 = None
        getitem_1003: "bf16[1024, 256, 1, 1]" = convolution_backward_53[1];  convolution_backward_53 = None
        convert_element_type_197: "f32[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1003, torch.float32);  getitem_1003 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_127 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 756, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_333, 'Y_ptr': full_default_342, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_333 = triton_kernel_wrapper_mutation_127 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_2378: "i8[512, 98, 512]" = torch.ops.aten.reshape.default(full_default_342, [512, 98, 512]);  full_default_342 = None
        view_2379: "i8[512, 256, 14, 14]" = torch.ops.aten.reshape.default(view_2378, [512, 256, 14, 14]);  view_2378 = None
        mul_396: "bf16[512, 256, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_999, view_2379);  getitem_999 = view_2379 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_612: "bf16[50176, 512]" = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_126 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 544, constant_args_idx = 757, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_328, 'S_ptr': getitem_329, 'M_ptr': getitem_330, 'Y_ptr': empty_612, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_328 = getitem_329 = getitem_330 = triton_kernel_wrapper_mutation_126 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_2394: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(mul_396, [512, 256, 196]);  mul_396 = None
        view_2395: "bf16[512, 98, 512]" = torch.ops.aten.reshape.default(empty_612, [512, 98, 512]);  empty_612 = None
        view_2396: "bf16[512, 256, 196]" = torch.ops.aten.reshape.default(view_2395, [512, 256, 196]);  view_2395 = None
        
        # No stacktrace found for following nodes
        as_strided_default_130: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_65: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_130);  as_strided_default_130 = None
        as_strided_default_131: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_65, [256], [1], 0);  clone_default_65 = None
        as_strided_default_132: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_66: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_132);  as_strided_default_132 = None
        as_strided_default_133: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_66, [256], [1], 0);  clone_default_66 = None
        triton_kernel_wrapper_mutation_125 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 545, constant_args_idx = 758, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2396, 'DY': view_2394, 'DBETA': as_strided_default_131, 'DGAMMA': as_strided_default_133, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_125 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_613: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_184: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_613, [0, 1, 2]);  empty_613 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_124 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 546, constant_args_idx = 759, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2396, 'DY': view_2394, 'INVSTD': rsqrt_25, 'GAMMA': primals_154, 'DBETA': as_strided_default_131, 'DGAMMA': as_strided_default_133, 'DX': permute_184, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024});  view_2396 = view_2394 = rsqrt_25 = primals_154 = triton_kernel_wrapper_mutation_124 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_614: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_123 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 547, constant_args_idx = 760, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_321, 'S_ptr': getitem_322, 'M_ptr': getitem_323, 'Y_ptr': empty_614, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_321 = getitem_322 = getitem_323 = triton_kernel_wrapper_mutation_123 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_2412: "bf16[512, 256, 14, 14]" = torch.ops.aten.reshape.default(permute_184, [512, 256, 14, 14]);  permute_184 = None
        empty_615: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_55: "bf16[512, 256, 28, 28]" = torch.ops.aten.expand.default(empty_615, [512, 256, 28, 28]);  empty_615 = None
        convolution_backward_54 = torch.ops.aten.convolution_backward.default(view_2412, expand_55, convert_element_type_26, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_55 = convert_element_type_26 = None
        getitem_1011: "bf16[512, 256, 28, 28]" = convolution_backward_54[0];  convolution_backward_54 = None
        empty_616: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_56: "bf16[256, 256, 3, 3]" = torch.ops.aten.expand.default(empty_616, [256, 256, 3, 3]);  empty_616 = None
        view_2413: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_614, [512, 392, 512]);  empty_614 = None
        view_2414: "bf16[512, 256, 28, 28]" = torch.ops.aten.reshape.default(view_2413, [512, 256, 28, 28]);  view_2413 = None
        convolution_backward_55 = torch.ops.aten.convolution_backward.default(view_2412, view_2414, expand_56, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_2412 = view_2414 = expand_56 = None
        getitem_1015: "bf16[256, 256, 3, 3]" = convolution_backward_55[1];  convolution_backward_55 = None
        convert_element_type_202: "f32[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_1015, torch.float32);  getitem_1015 = None
        
        # No stacktrace found for following nodes
        as_strided_default_128: "i8[102760448]" = torch.ops.aten.as_strided.default(full_default_339, [102760448], [1], 0)
        clone_default_64: "i8[102760448]" = torch.ops.aten.clone.default(as_strided_default_128);  as_strided_default_128 = None
        as_strided_default_129: "i8[200704, 512]" = torch.ops.aten.as_strided.default(clone_default_64, [200704, 512], [512, 1], 0);  clone_default_64 = None
        triton_kernel_wrapper_mutation_122 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 761, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_320, 'Y_ptr': as_strided_default_129, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_320 = triton_kernel_wrapper_mutation_122 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_2417: "i8[512, 392, 512]" = torch.ops.aten.reshape.default(as_strided_default_129, [512, 392, 512]);  as_strided_default_129 = None
        view_2418: "i8[512, 256, 28, 28]" = torch.ops.aten.reshape.default(view_2417, [512, 256, 28, 28]);  view_2417 = None
        mul_397: "bf16[512, 256, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_1011, view_2418);  getitem_1011 = view_2418 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_617: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_121 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 548, constant_args_idx = 762, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_315, 'S_ptr': getitem_316, 'M_ptr': getitem_317, 'Y_ptr': empty_617, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_315 = getitem_316 = getitem_317 = triton_kernel_wrapper_mutation_121 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_2433: "bf16[512, 256, 784]" = torch.ops.aten.reshape.default(mul_397, [512, 256, 784]);  mul_397 = None
        view_2434: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_617, [512, 392, 512]);  empty_617 = None
        view_2435: "bf16[512, 256, 784]" = torch.ops.aten.reshape.default(view_2434, [512, 256, 784]);  view_2434 = None
        
        # No stacktrace found for following nodes
        as_strided_default_124: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_62: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_124);  as_strided_default_124 = None
        as_strided_default_125: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_62, [256], [1], 0);  clone_default_62 = None
        as_strided_default_126: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_63: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_126);  as_strided_default_126 = None
        as_strided_default_127: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_63, [256], [1], 0);  clone_default_63 = None
        triton_kernel_wrapper_mutation_120 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 549, constant_args_idx = 763, grid = [(256, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2435, 'DY': view_2433, 'DBETA': as_strided_default_125, 'DGAMMA': as_strided_default_127, 'M': 401408, 'HW': 784, 'stride_n': 200704, 'stride_c': 784, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_120 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_618: "bf16[512, 256, 784]" = torch.ops.aten.empty.memory_format([512, 256, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_185: "bf16[512, 256, 784]" = torch.ops.aten.permute.default(empty_618, [0, 1, 2]);  empty_618 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_119 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 550, constant_args_idx = 764, grid = [(256, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2435, 'DY': view_2433, 'INVSTD': rsqrt_24, 'GAMMA': primals_148, 'DBETA': as_strided_default_125, 'DGAMMA': as_strided_default_127, 'DX': permute_185, 'M': 401408, 'HW': 784, 'stride_n': 200704, 'stride_c': 784, 'BLOCK_M': 1024});  view_2435 = view_2433 = rsqrt_24 = primals_148 = triton_kernel_wrapper_mutation_119 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_619: "bf16[401408, 512]" = torch.ops.aten.empty.memory_format([401408, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_118 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 551, constant_args_idx = 765, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_308, 'S_ptr': getitem_309, 'M_ptr': getitem_310, 'Y_ptr': empty_619, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_308 = getitem_309 = getitem_310 = triton_kernel_wrapper_mutation_118 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_2451: "bf16[512, 256, 28, 28]" = torch.ops.aten.reshape.default(permute_185, [512, 256, 28, 28]);  permute_185 = None
        empty_620: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_57: "bf16[512, 512, 28, 28]" = torch.ops.aten.expand.default(empty_620, [512, 512, 28, 28]);  empty_620 = None
        convolution_backward_56 = torch.ops.aten.convolution_backward.default(view_2451, expand_57, convert_element_type_25, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_57 = convert_element_type_25 = None
        getitem_1023: "bf16[512, 512, 28, 28]" = convolution_backward_56[0];  convolution_backward_56 = None
        empty_621: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_58: "bf16[256, 512, 1, 1]" = torch.ops.aten.expand.default(empty_621, [256, 512, 1, 1]);  empty_621 = None
        view_2452: "bf16[512, 784, 512]" = torch.ops.aten.reshape.default(empty_619, [512, 784, 512]);  empty_619 = None
        view_2453: "bf16[512, 512, 28, 28]" = torch.ops.aten.reshape.default(view_2452, [512, 512, 28, 28]);  view_2452 = None
        convolution_backward_57 = torch.ops.aten.convolution_backward.default(view_2451, view_2453, expand_58, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2451 = view_2453 = expand_58 = None
        getitem_1027: "bf16[256, 512, 1, 1]" = convolution_backward_57[1];  convolution_backward_57 = None
        convert_element_type_207: "f32[256, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1027, torch.float32);  getitem_1027 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_236: "bf16[512, 512, 28, 28]" = torch.ops.aten.add.Tensor(getitem_988, getitem_1023);  getitem_988 = getitem_1023 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_395: "i8[401408, 512]" = torch.ops.aten.full.default([401408, 512], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        as_strided_default_122: "i8[205520896]" = torch.ops.aten.as_strided.default(full_default_395, [205520896], [1], 0)
        clone_default_61: "i8[205520896]" = torch.ops.aten.clone.default(as_strided_default_122);  as_strided_default_122 = None
        as_strided_default_123: "i8[401408, 512]" = torch.ops.aten.as_strided.default(clone_default_61, [401408, 512], [512, 1], 0);  clone_default_61 = None
        triton_kernel_wrapper_mutation_117 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 766, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_307, 'Y_ptr': as_strided_default_123, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_307 = triton_kernel_wrapper_mutation_117 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_2456: "i8[512, 784, 512]" = torch.ops.aten.reshape.default(as_strided_default_123, [512, 784, 512]);  as_strided_default_123 = None
        view_2457: "i8[512, 512, 28, 28]" = torch.ops.aten.reshape.default(view_2456, [512, 512, 28, 28]);  view_2456 = None
        mul_398: "bf16[512, 512, 28, 28]" = torch.ops.aten.mul.Tensor(add_236, view_2457);  add_236 = view_2457 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_622: "bf16[401408, 512]" = torch.ops.aten.empty.memory_format([401408, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_116 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 552, constant_args_idx = 767, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_302, 'S_ptr': getitem_303, 'M_ptr': getitem_304, 'Y_ptr': empty_622, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_302 = getitem_303 = getitem_304 = triton_kernel_wrapper_mutation_116 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_2472: "bf16[512, 512, 784]" = torch.ops.aten.reshape.default(mul_398, [512, 512, 784])
        view_2473: "bf16[512, 784, 512]" = torch.ops.aten.reshape.default(empty_622, [512, 784, 512]);  empty_622 = None
        view_2474: "bf16[512, 512, 784]" = torch.ops.aten.reshape.default(view_2473, [512, 512, 784]);  view_2473 = None
        
        # No stacktrace found for following nodes
        as_strided_default_118: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_59: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_118);  as_strided_default_118 = None
        as_strided_default_119: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_59, [512], [1], 0);  clone_default_59 = None
        as_strided_default_120: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_60: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_120);  as_strided_default_120 = None
        as_strided_default_121: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_60, [512], [1], 0);  clone_default_60 = None
        triton_kernel_wrapper_mutation_115 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 553, constant_args_idx = 768, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2474, 'DY': view_2472, 'DBETA': as_strided_default_119, 'DGAMMA': as_strided_default_121, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_115 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_623: "bf16[512, 512, 784]" = torch.ops.aten.empty.memory_format([512, 512, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_186: "bf16[512, 512, 784]" = torch.ops.aten.permute.default(empty_623, [0, 1, 2]);  empty_623 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_114 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 554, constant_args_idx = 769, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2474, 'DY': view_2472, 'INVSTD': rsqrt_23, 'GAMMA': primals_142, 'DBETA': as_strided_default_119, 'DGAMMA': as_strided_default_121, 'DX': permute_186, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024});  view_2474 = view_2472 = rsqrt_23 = primals_142 = triton_kernel_wrapper_mutation_114 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_624: "bf16[100352, 512]" = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_113 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 555, constant_args_idx = 770, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_295, 'S_ptr': getitem_296, 'M_ptr': getitem_297, 'Y_ptr': empty_624, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_295 = getitem_296 = getitem_297 = triton_kernel_wrapper_mutation_113 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_2490: "bf16[512, 512, 28, 28]" = torch.ops.aten.reshape.default(permute_186, [512, 512, 28, 28]);  permute_186 = None
        empty_625: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_59: "bf16[512, 128, 28, 28]" = torch.ops.aten.expand.default(empty_625, [512, 128, 28, 28]);  empty_625 = None
        convolution_backward_58 = torch.ops.aten.convolution_backward.default(view_2490, expand_59, convert_element_type_24, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_59 = convert_element_type_24 = None
        getitem_1035: "bf16[512, 128, 28, 28]" = convolution_backward_58[0];  convolution_backward_58 = None
        empty_626: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_60: "bf16[512, 128, 1, 1]" = torch.ops.aten.expand.default(empty_626, [512, 128, 1, 1]);  empty_626 = None
        view_2491: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(empty_624, [512, 196, 512]);  empty_624 = None
        view_2492: "bf16[512, 128, 28, 28]" = torch.ops.aten.reshape.default(view_2491, [512, 128, 28, 28]);  view_2491 = None
        convolution_backward_59 = torch.ops.aten.convolution_backward.default(view_2490, view_2492, expand_60, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2490 = view_2492 = expand_60 = None
        getitem_1039: "bf16[512, 128, 1, 1]" = convolution_backward_59[1];  convolution_backward_59 = None
        convert_element_type_212: "f32[512, 128, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1039, torch.float32);  getitem_1039 = None
        
        # No stacktrace found for following nodes
        as_strided_default_116: "i8[51380224]" = torch.ops.aten.as_strided.default(full_default_310, [51380224], [1], 0)
        clone_default_58: "i8[51380224]" = torch.ops.aten.clone.default(as_strided_default_116);  as_strided_default_116 = None
        as_strided_default_117: "i8[100352, 512]" = torch.ops.aten.as_strided.default(clone_default_58, [100352, 512], [512, 1], 0);  clone_default_58 = None
        triton_kernel_wrapper_mutation_112 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 771, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_294, 'Y_ptr': as_strided_default_117, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_294 = triton_kernel_wrapper_mutation_112 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_2495: "i8[512, 196, 512]" = torch.ops.aten.reshape.default(as_strided_default_117, [512, 196, 512]);  as_strided_default_117 = None
        view_2496: "i8[512, 128, 28, 28]" = torch.ops.aten.reshape.default(view_2495, [512, 128, 28, 28]);  view_2495 = None
        mul_399: "bf16[512, 128, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_1035, view_2496);  getitem_1035 = view_2496 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_627: "bf16[100352, 512]" = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_111 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 556, constant_args_idx = 772, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_289, 'S_ptr': getitem_290, 'M_ptr': getitem_291, 'Y_ptr': empty_627, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_289 = getitem_290 = getitem_291 = triton_kernel_wrapper_mutation_111 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_2511: "bf16[512, 128, 784]" = torch.ops.aten.reshape.default(mul_399, [512, 128, 784]);  mul_399 = None
        view_2512: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(empty_627, [512, 196, 512]);  empty_627 = None
        view_2513: "bf16[512, 128, 784]" = torch.ops.aten.reshape.default(view_2512, [512, 128, 784]);  view_2512 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        full_default_64: "f32[128]" = torch.ops.aten.full.default([128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        as_strided_default_112: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_56: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_112);  as_strided_default_112 = None
        as_strided_default_113: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_56, [128], [1], 0);  clone_default_56 = None
        as_strided_default_114: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_57: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_114);  as_strided_default_114 = None
        as_strided_default_115: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_57, [128], [1], 0);  clone_default_57 = None
        triton_kernel_wrapper_mutation_110 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 557, constant_args_idx = 773, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2513, 'DY': view_2511, 'DBETA': as_strided_default_113, 'DGAMMA': as_strided_default_115, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_110 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_628: "bf16[512, 128, 784]" = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_187: "bf16[512, 128, 784]" = torch.ops.aten.permute.default(empty_628, [0, 1, 2]);  empty_628 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_109 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 558, constant_args_idx = 774, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2513, 'DY': view_2511, 'INVSTD': rsqrt_22, 'GAMMA': primals_136, 'DBETA': as_strided_default_113, 'DGAMMA': as_strided_default_115, 'DX': permute_187, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024});  view_2513 = view_2511 = rsqrt_22 = primals_136 = triton_kernel_wrapper_mutation_109 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_629: "bf16[100352, 512]" = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_108 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 559, constant_args_idx = 775, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_282, 'S_ptr': getitem_283, 'M_ptr': getitem_284, 'Y_ptr': empty_629, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_282 = getitem_283 = getitem_284 = triton_kernel_wrapper_mutation_108 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_2529: "bf16[512, 128, 28, 28]" = torch.ops.aten.reshape.default(permute_187, [512, 128, 28, 28]);  permute_187 = None
        empty_630: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_61: "bf16[512, 128, 28, 28]" = torch.ops.aten.expand.default(empty_630, [512, 128, 28, 28]);  empty_630 = None
        convolution_backward_60 = torch.ops.aten.convolution_backward.default(view_2529, expand_61, convert_element_type_23, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_61 = convert_element_type_23 = None
        getitem_1047: "bf16[512, 128, 28, 28]" = convolution_backward_60[0];  convolution_backward_60 = None
        empty_631: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_62: "bf16[128, 128, 3, 3]" = torch.ops.aten.expand.default(empty_631, [128, 128, 3, 3]);  empty_631 = None
        view_2530: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(empty_629, [512, 196, 512]);  empty_629 = None
        view_2531: "bf16[512, 128, 28, 28]" = torch.ops.aten.reshape.default(view_2530, [512, 128, 28, 28]);  view_2530 = None
        convolution_backward_61 = torch.ops.aten.convolution_backward.default(view_2529, view_2531, expand_62, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_2529 = view_2531 = expand_62 = None
        getitem_1051: "bf16[128, 128, 3, 3]" = convolution_backward_61[1];  convolution_backward_61 = None
        convert_element_type_217: "f32[128, 128, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_1051, torch.float32);  getitem_1051 = None
        
        # No stacktrace found for following nodes
        as_strided_default_110: "i8[51380224]" = torch.ops.aten.as_strided.default(full_default_310, [51380224], [1], 0)
        clone_default_55: "i8[51380224]" = torch.ops.aten.clone.default(as_strided_default_110);  as_strided_default_110 = None
        as_strided_default_111: "i8[100352, 512]" = torch.ops.aten.as_strided.default(clone_default_55, [100352, 512], [512, 1], 0);  clone_default_55 = None
        triton_kernel_wrapper_mutation_107 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 776, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_281, 'Y_ptr': as_strided_default_111, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_281 = triton_kernel_wrapper_mutation_107 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_2534: "i8[512, 196, 512]" = torch.ops.aten.reshape.default(as_strided_default_111, [512, 196, 512]);  as_strided_default_111 = None
        view_2535: "i8[512, 128, 28, 28]" = torch.ops.aten.reshape.default(view_2534, [512, 128, 28, 28]);  view_2534 = None
        mul_400: "bf16[512, 128, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_1047, view_2535);  getitem_1047 = view_2535 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_632: "bf16[100352, 512]" = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_106 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 560, constant_args_idx = 777, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_276, 'S_ptr': getitem_277, 'M_ptr': getitem_278, 'Y_ptr': empty_632, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_276 = getitem_277 = getitem_278 = triton_kernel_wrapper_mutation_106 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_2550: "bf16[512, 128, 784]" = torch.ops.aten.reshape.default(mul_400, [512, 128, 784]);  mul_400 = None
        view_2551: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(empty_632, [512, 196, 512]);  empty_632 = None
        view_2552: "bf16[512, 128, 784]" = torch.ops.aten.reshape.default(view_2551, [512, 128, 784]);  view_2551 = None
        
        # No stacktrace found for following nodes
        as_strided_default_106: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_53: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_106);  as_strided_default_106 = None
        as_strided_default_107: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_53, [128], [1], 0);  clone_default_53 = None
        as_strided_default_108: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_54: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_108);  as_strided_default_108 = None
        as_strided_default_109: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_54, [128], [1], 0);  clone_default_54 = None
        triton_kernel_wrapper_mutation_105 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 561, constant_args_idx = 778, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2552, 'DY': view_2550, 'DBETA': as_strided_default_107, 'DGAMMA': as_strided_default_109, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_105 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_633: "bf16[512, 128, 784]" = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_188: "bf16[512, 128, 784]" = torch.ops.aten.permute.default(empty_633, [0, 1, 2]);  empty_633 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_104 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 562, constant_args_idx = 779, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2552, 'DY': view_2550, 'INVSTD': rsqrt_21, 'GAMMA': primals_130, 'DBETA': as_strided_default_107, 'DGAMMA': as_strided_default_109, 'DX': permute_188, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024});  view_2552 = view_2550 = rsqrt_21 = primals_130 = triton_kernel_wrapper_mutation_104 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_634: "bf16[401408, 512]" = torch.ops.aten.empty.memory_format([401408, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_103 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 563, constant_args_idx = 780, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_269, 'S_ptr': getitem_270, 'M_ptr': getitem_271, 'Y_ptr': empty_634, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_269 = getitem_270 = getitem_271 = triton_kernel_wrapper_mutation_103 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_2568: "bf16[512, 128, 28, 28]" = torch.ops.aten.reshape.default(permute_188, [512, 128, 28, 28]);  permute_188 = None
        empty_635: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_63: "bf16[512, 512, 28, 28]" = torch.ops.aten.expand.default(empty_635, [512, 512, 28, 28]);  empty_635 = None
        convolution_backward_62 = torch.ops.aten.convolution_backward.default(view_2568, expand_63, convert_element_type_22, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_63 = convert_element_type_22 = None
        getitem_1059: "bf16[512, 512, 28, 28]" = convolution_backward_62[0];  convolution_backward_62 = None
        empty_636: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_64: "bf16[128, 512, 1, 1]" = torch.ops.aten.expand.default(empty_636, [128, 512, 1, 1]);  empty_636 = None
        view_2569: "bf16[512, 784, 512]" = torch.ops.aten.reshape.default(empty_634, [512, 784, 512]);  empty_634 = None
        view_2570: "bf16[512, 512, 28, 28]" = torch.ops.aten.reshape.default(view_2569, [512, 512, 28, 28]);  view_2569 = None
        convolution_backward_63 = torch.ops.aten.convolution_backward.default(view_2568, view_2570, expand_64, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2568 = view_2570 = expand_64 = None
        getitem_1063: "bf16[128, 512, 1, 1]" = convolution_backward_63[1];  convolution_backward_63 = None
        convert_element_type_222: "f32[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1063, torch.float32);  getitem_1063 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_237: "bf16[512, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_398, getitem_1059);  mul_398 = getitem_1059 = None
        
        # No stacktrace found for following nodes
        as_strided_default_104: "i8[205520896]" = torch.ops.aten.as_strided.default(full_default_395, [205520896], [1], 0)
        clone_default_52: "i8[205520896]" = torch.ops.aten.clone.default(as_strided_default_104);  as_strided_default_104 = None
        as_strided_default_105: "i8[401408, 512]" = torch.ops.aten.as_strided.default(clone_default_52, [401408, 512], [512, 1], 0);  clone_default_52 = None
        triton_kernel_wrapper_mutation_102 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 781, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_268, 'Y_ptr': as_strided_default_105, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_268 = triton_kernel_wrapper_mutation_102 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_2573: "i8[512, 784, 512]" = torch.ops.aten.reshape.default(as_strided_default_105, [512, 784, 512]);  as_strided_default_105 = None
        view_2574: "i8[512, 512, 28, 28]" = torch.ops.aten.reshape.default(view_2573, [512, 512, 28, 28]);  view_2573 = None
        mul_401: "bf16[512, 512, 28, 28]" = torch.ops.aten.mul.Tensor(add_237, view_2574);  add_237 = view_2574 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_637: "bf16[401408, 512]" = torch.ops.aten.empty.memory_format([401408, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_101 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 564, constant_args_idx = 782, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_263, 'S_ptr': getitem_264, 'M_ptr': getitem_265, 'Y_ptr': empty_637, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_263 = getitem_264 = getitem_265 = triton_kernel_wrapper_mutation_101 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_2589: "bf16[512, 512, 784]" = torch.ops.aten.reshape.default(mul_401, [512, 512, 784])
        view_2590: "bf16[512, 784, 512]" = torch.ops.aten.reshape.default(empty_637, [512, 784, 512]);  empty_637 = None
        view_2591: "bf16[512, 512, 784]" = torch.ops.aten.reshape.default(view_2590, [512, 512, 784]);  view_2590 = None
        
        # No stacktrace found for following nodes
        as_strided_default_100: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_50: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_100);  as_strided_default_100 = None
        as_strided_default_101: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_50, [512], [1], 0);  clone_default_50 = None
        as_strided_default_102: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_51: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_102);  as_strided_default_102 = None
        as_strided_default_103: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_51, [512], [1], 0);  clone_default_51 = None
        triton_kernel_wrapper_mutation_100 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 565, constant_args_idx = 783, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2591, 'DY': view_2589, 'DBETA': as_strided_default_101, 'DGAMMA': as_strided_default_103, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_100 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_638: "bf16[512, 512, 784]" = torch.ops.aten.empty.memory_format([512, 512, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_189: "bf16[512, 512, 784]" = torch.ops.aten.permute.default(empty_638, [0, 1, 2]);  empty_638 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_99 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 566, constant_args_idx = 784, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2591, 'DY': view_2589, 'INVSTD': rsqrt_20, 'GAMMA': primals_124, 'DBETA': as_strided_default_101, 'DGAMMA': as_strided_default_103, 'DX': permute_189, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024});  view_2591 = view_2589 = rsqrt_20 = primals_124 = triton_kernel_wrapper_mutation_99 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_639: "bf16[100352, 512]" = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_98 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 567, constant_args_idx = 785, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_256, 'S_ptr': getitem_257, 'M_ptr': getitem_258, 'Y_ptr': empty_639, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_256 = getitem_257 = getitem_258 = triton_kernel_wrapper_mutation_98 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_2607: "bf16[512, 512, 28, 28]" = torch.ops.aten.reshape.default(permute_189, [512, 512, 28, 28]);  permute_189 = None
        empty_640: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_65: "bf16[512, 128, 28, 28]" = torch.ops.aten.expand.default(empty_640, [512, 128, 28, 28]);  empty_640 = None
        convolution_backward_64 = torch.ops.aten.convolution_backward.default(view_2607, expand_65, convert_element_type_21, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_65 = convert_element_type_21 = None
        getitem_1071: "bf16[512, 128, 28, 28]" = convolution_backward_64[0];  convolution_backward_64 = None
        empty_641: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_66: "bf16[512, 128, 1, 1]" = torch.ops.aten.expand.default(empty_641, [512, 128, 1, 1]);  empty_641 = None
        view_2608: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(empty_639, [512, 196, 512]);  empty_639 = None
        view_2609: "bf16[512, 128, 28, 28]" = torch.ops.aten.reshape.default(view_2608, [512, 128, 28, 28]);  view_2608 = None
        convolution_backward_65 = torch.ops.aten.convolution_backward.default(view_2607, view_2609, expand_66, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2607 = view_2609 = expand_66 = None
        getitem_1075: "bf16[512, 128, 1, 1]" = convolution_backward_65[1];  convolution_backward_65 = None
        convert_element_type_227: "f32[512, 128, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1075, torch.float32);  getitem_1075 = None
        
        # No stacktrace found for following nodes
        as_strided_default_98: "i8[51380224]" = torch.ops.aten.as_strided.default(full_default_310, [51380224], [1], 0)
        clone_default_49: "i8[51380224]" = torch.ops.aten.clone.default(as_strided_default_98);  as_strided_default_98 = None
        as_strided_default_99: "i8[100352, 512]" = torch.ops.aten.as_strided.default(clone_default_49, [100352, 512], [512, 1], 0);  clone_default_49 = None
        triton_kernel_wrapper_mutation_97 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 786, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_255, 'Y_ptr': as_strided_default_99, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_255 = triton_kernel_wrapper_mutation_97 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_2612: "i8[512, 196, 512]" = torch.ops.aten.reshape.default(as_strided_default_99, [512, 196, 512]);  as_strided_default_99 = None
        view_2613: "i8[512, 128, 28, 28]" = torch.ops.aten.reshape.default(view_2612, [512, 128, 28, 28]);  view_2612 = None
        mul_402: "bf16[512, 128, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_1071, view_2613);  getitem_1071 = view_2613 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_642: "bf16[100352, 512]" = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_96 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 568, constant_args_idx = 787, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_250, 'S_ptr': getitem_251, 'M_ptr': getitem_252, 'Y_ptr': empty_642, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_250 = getitem_251 = getitem_252 = triton_kernel_wrapper_mutation_96 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_2628: "bf16[512, 128, 784]" = torch.ops.aten.reshape.default(mul_402, [512, 128, 784]);  mul_402 = None
        view_2629: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(empty_642, [512, 196, 512]);  empty_642 = None
        view_2630: "bf16[512, 128, 784]" = torch.ops.aten.reshape.default(view_2629, [512, 128, 784]);  view_2629 = None
        
        # No stacktrace found for following nodes
        as_strided_default_94: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_47: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_94);  as_strided_default_94 = None
        as_strided_default_95: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_47, [128], [1], 0);  clone_default_47 = None
        as_strided_default_96: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_48: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_96);  as_strided_default_96 = None
        as_strided_default_97: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_48, [128], [1], 0);  clone_default_48 = None
        triton_kernel_wrapper_mutation_95 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 569, constant_args_idx = 788, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2630, 'DY': view_2628, 'DBETA': as_strided_default_95, 'DGAMMA': as_strided_default_97, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_95 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_643: "bf16[512, 128, 784]" = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_190: "bf16[512, 128, 784]" = torch.ops.aten.permute.default(empty_643, [0, 1, 2]);  empty_643 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_94 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 570, constant_args_idx = 789, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2630, 'DY': view_2628, 'INVSTD': rsqrt_19, 'GAMMA': primals_118, 'DBETA': as_strided_default_95, 'DGAMMA': as_strided_default_97, 'DX': permute_190, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024});  view_2630 = view_2628 = rsqrt_19 = primals_118 = triton_kernel_wrapper_mutation_94 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_644: "bf16[100352, 512]" = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_93 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 571, constant_args_idx = 790, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_243, 'S_ptr': getitem_244, 'M_ptr': getitem_245, 'Y_ptr': empty_644, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_243 = getitem_244 = getitem_245 = triton_kernel_wrapper_mutation_93 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_2646: "bf16[512, 128, 28, 28]" = torch.ops.aten.reshape.default(permute_190, [512, 128, 28, 28]);  permute_190 = None
        empty_645: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_67: "bf16[512, 128, 28, 28]" = torch.ops.aten.expand.default(empty_645, [512, 128, 28, 28]);  empty_645 = None
        convolution_backward_66 = torch.ops.aten.convolution_backward.default(view_2646, expand_67, convert_element_type_20, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_67 = convert_element_type_20 = None
        getitem_1083: "bf16[512, 128, 28, 28]" = convolution_backward_66[0];  convolution_backward_66 = None
        empty_646: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_68: "bf16[128, 128, 3, 3]" = torch.ops.aten.expand.default(empty_646, [128, 128, 3, 3]);  empty_646 = None
        view_2647: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(empty_644, [512, 196, 512]);  empty_644 = None
        view_2648: "bf16[512, 128, 28, 28]" = torch.ops.aten.reshape.default(view_2647, [512, 128, 28, 28]);  view_2647 = None
        convolution_backward_67 = torch.ops.aten.convolution_backward.default(view_2646, view_2648, expand_68, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_2646 = view_2648 = expand_68 = None
        getitem_1087: "bf16[128, 128, 3, 3]" = convolution_backward_67[1];  convolution_backward_67 = None
        convert_element_type_232: "f32[128, 128, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_1087, torch.float32);  getitem_1087 = None
        
        # No stacktrace found for following nodes
        as_strided_default_92: "i8[51380224]" = torch.ops.aten.as_strided.default(full_default_310, [51380224], [1], 0)
        clone_default_46: "i8[51380224]" = torch.ops.aten.clone.default(as_strided_default_92);  as_strided_default_92 = None
        as_strided_default_93: "i8[100352, 512]" = torch.ops.aten.as_strided.default(clone_default_46, [100352, 512], [512, 1], 0);  clone_default_46 = None
        triton_kernel_wrapper_mutation_92 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 791, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_242, 'Y_ptr': as_strided_default_93, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_242 = triton_kernel_wrapper_mutation_92 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_2651: "i8[512, 196, 512]" = torch.ops.aten.reshape.default(as_strided_default_93, [512, 196, 512]);  as_strided_default_93 = None
        view_2652: "i8[512, 128, 28, 28]" = torch.ops.aten.reshape.default(view_2651, [512, 128, 28, 28]);  view_2651 = None
        mul_403: "bf16[512, 128, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_1083, view_2652);  getitem_1083 = view_2652 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_647: "bf16[100352, 512]" = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_91 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 572, constant_args_idx = 792, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_237, 'S_ptr': getitem_238, 'M_ptr': getitem_239, 'Y_ptr': empty_647, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_237 = getitem_238 = getitem_239 = triton_kernel_wrapper_mutation_91 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_2667: "bf16[512, 128, 784]" = torch.ops.aten.reshape.default(mul_403, [512, 128, 784]);  mul_403 = None
        view_2668: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(empty_647, [512, 196, 512]);  empty_647 = None
        view_2669: "bf16[512, 128, 784]" = torch.ops.aten.reshape.default(view_2668, [512, 128, 784]);  view_2668 = None
        
        # No stacktrace found for following nodes
        as_strided_default_88: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_44: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_88);  as_strided_default_88 = None
        as_strided_default_89: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_44, [128], [1], 0);  clone_default_44 = None
        as_strided_default_90: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_45: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_90);  as_strided_default_90 = None
        as_strided_default_91: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_45, [128], [1], 0);  clone_default_45 = None
        triton_kernel_wrapper_mutation_90 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 573, constant_args_idx = 793, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2669, 'DY': view_2667, 'DBETA': as_strided_default_89, 'DGAMMA': as_strided_default_91, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_90 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_648: "bf16[512, 128, 784]" = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_191: "bf16[512, 128, 784]" = torch.ops.aten.permute.default(empty_648, [0, 1, 2]);  empty_648 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_89 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 574, constant_args_idx = 794, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2669, 'DY': view_2667, 'INVSTD': rsqrt_18, 'GAMMA': primals_112, 'DBETA': as_strided_default_89, 'DGAMMA': as_strided_default_91, 'DX': permute_191, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024});  view_2669 = view_2667 = rsqrt_18 = primals_112 = triton_kernel_wrapper_mutation_89 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_649: "bf16[401408, 512]" = torch.ops.aten.empty.memory_format([401408, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_88 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 575, constant_args_idx = 795, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_230, 'S_ptr': getitem_231, 'M_ptr': getitem_232, 'Y_ptr': empty_649, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_230 = getitem_231 = getitem_232 = triton_kernel_wrapper_mutation_88 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_2685: "bf16[512, 128, 28, 28]" = torch.ops.aten.reshape.default(permute_191, [512, 128, 28, 28]);  permute_191 = None
        empty_650: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_69: "bf16[512, 512, 28, 28]" = torch.ops.aten.expand.default(empty_650, [512, 512, 28, 28]);  empty_650 = None
        convolution_backward_68 = torch.ops.aten.convolution_backward.default(view_2685, expand_69, convert_element_type_19, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_69 = convert_element_type_19 = None
        getitem_1095: "bf16[512, 512, 28, 28]" = convolution_backward_68[0];  convolution_backward_68 = None
        empty_651: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_70: "bf16[128, 512, 1, 1]" = torch.ops.aten.expand.default(empty_651, [128, 512, 1, 1]);  empty_651 = None
        view_2686: "bf16[512, 784, 512]" = torch.ops.aten.reshape.default(empty_649, [512, 784, 512]);  empty_649 = None
        view_2687: "bf16[512, 512, 28, 28]" = torch.ops.aten.reshape.default(view_2686, [512, 512, 28, 28]);  view_2686 = None
        convolution_backward_69 = torch.ops.aten.convolution_backward.default(view_2685, view_2687, expand_70, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2685 = view_2687 = expand_70 = None
        getitem_1099: "bf16[128, 512, 1, 1]" = convolution_backward_69[1];  convolution_backward_69 = None
        convert_element_type_237: "f32[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1099, torch.float32);  getitem_1099 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_238: "bf16[512, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_401, getitem_1095);  mul_401 = getitem_1095 = None
        
        # No stacktrace found for following nodes
        as_strided_default_86: "i8[205520896]" = torch.ops.aten.as_strided.default(full_default_395, [205520896], [1], 0)
        clone_default_43: "i8[205520896]" = torch.ops.aten.clone.default(as_strided_default_86);  as_strided_default_86 = None
        as_strided_default_87: "i8[401408, 512]" = torch.ops.aten.as_strided.default(clone_default_43, [401408, 512], [512, 1], 0);  clone_default_43 = None
        triton_kernel_wrapper_mutation_87 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 796, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_229, 'Y_ptr': as_strided_default_87, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_229 = triton_kernel_wrapper_mutation_87 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_2690: "i8[512, 784, 512]" = torch.ops.aten.reshape.default(as_strided_default_87, [512, 784, 512]);  as_strided_default_87 = None
        view_2691: "i8[512, 512, 28, 28]" = torch.ops.aten.reshape.default(view_2690, [512, 512, 28, 28]);  view_2690 = None
        mul_404: "bf16[512, 512, 28, 28]" = torch.ops.aten.mul.Tensor(add_238, view_2691);  add_238 = view_2691 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_652: "bf16[401408, 512]" = torch.ops.aten.empty.memory_format([401408, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_86 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 576, constant_args_idx = 797, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_224, 'S_ptr': getitem_225, 'M_ptr': getitem_226, 'Y_ptr': empty_652, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_224 = getitem_225 = getitem_226 = triton_kernel_wrapper_mutation_86 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_2706: "bf16[512, 512, 784]" = torch.ops.aten.reshape.default(mul_404, [512, 512, 784])
        view_2707: "bf16[512, 784, 512]" = torch.ops.aten.reshape.default(empty_652, [512, 784, 512]);  empty_652 = None
        view_2708: "bf16[512, 512, 784]" = torch.ops.aten.reshape.default(view_2707, [512, 512, 784]);  view_2707 = None
        
        # No stacktrace found for following nodes
        as_strided_default_82: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_41: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_82);  as_strided_default_82 = None
        as_strided_default_83: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_41, [512], [1], 0);  clone_default_41 = None
        as_strided_default_84: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_42: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_84);  as_strided_default_84 = None
        as_strided_default_85: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_42, [512], [1], 0);  clone_default_42 = None
        triton_kernel_wrapper_mutation_85 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 577, constant_args_idx = 798, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2708, 'DY': view_2706, 'DBETA': as_strided_default_83, 'DGAMMA': as_strided_default_85, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_85 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_653: "bf16[512, 512, 784]" = torch.ops.aten.empty.memory_format([512, 512, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_192: "bf16[512, 512, 784]" = torch.ops.aten.permute.default(empty_653, [0, 1, 2]);  empty_653 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_84 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 578, constant_args_idx = 799, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2708, 'DY': view_2706, 'INVSTD': rsqrt_17, 'GAMMA': primals_106, 'DBETA': as_strided_default_83, 'DGAMMA': as_strided_default_85, 'DX': permute_192, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024});  view_2708 = view_2706 = rsqrt_17 = primals_106 = triton_kernel_wrapper_mutation_84 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_654: "bf16[100352, 512]" = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_83 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 579, constant_args_idx = 800, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_217, 'S_ptr': getitem_218, 'M_ptr': getitem_219, 'Y_ptr': empty_654, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_217 = getitem_218 = getitem_219 = triton_kernel_wrapper_mutation_83 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_2724: "bf16[512, 512, 28, 28]" = torch.ops.aten.reshape.default(permute_192, [512, 512, 28, 28]);  permute_192 = None
        empty_655: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_71: "bf16[512, 128, 28, 28]" = torch.ops.aten.expand.default(empty_655, [512, 128, 28, 28]);  empty_655 = None
        convolution_backward_70 = torch.ops.aten.convolution_backward.default(view_2724, expand_71, convert_element_type_18, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_71 = convert_element_type_18 = None
        getitem_1107: "bf16[512, 128, 28, 28]" = convolution_backward_70[0];  convolution_backward_70 = None
        empty_656: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_72: "bf16[512, 128, 1, 1]" = torch.ops.aten.expand.default(empty_656, [512, 128, 1, 1]);  empty_656 = None
        view_2725: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(empty_654, [512, 196, 512]);  empty_654 = None
        view_2726: "bf16[512, 128, 28, 28]" = torch.ops.aten.reshape.default(view_2725, [512, 128, 28, 28]);  view_2725 = None
        convolution_backward_71 = torch.ops.aten.convolution_backward.default(view_2724, view_2726, expand_72, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2724 = view_2726 = expand_72 = None
        getitem_1111: "bf16[512, 128, 1, 1]" = convolution_backward_71[1];  convolution_backward_71 = None
        convert_element_type_242: "f32[512, 128, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1111, torch.float32);  getitem_1111 = None
        
        # No stacktrace found for following nodes
        as_strided_default_80: "i8[51380224]" = torch.ops.aten.as_strided.default(full_default_310, [51380224], [1], 0)
        clone_default_40: "i8[51380224]" = torch.ops.aten.clone.default(as_strided_default_80);  as_strided_default_80 = None
        as_strided_default_81: "i8[100352, 512]" = torch.ops.aten.as_strided.default(clone_default_40, [100352, 512], [512, 1], 0);  clone_default_40 = None
        triton_kernel_wrapper_mutation_82 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 801, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_216, 'Y_ptr': as_strided_default_81, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_216 = triton_kernel_wrapper_mutation_82 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_2729: "i8[512, 196, 512]" = torch.ops.aten.reshape.default(as_strided_default_81, [512, 196, 512]);  as_strided_default_81 = None
        view_2730: "i8[512, 128, 28, 28]" = torch.ops.aten.reshape.default(view_2729, [512, 128, 28, 28]);  view_2729 = None
        mul_405: "bf16[512, 128, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_1107, view_2730);  getitem_1107 = view_2730 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_657: "bf16[100352, 512]" = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_81 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 580, constant_args_idx = 802, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_211, 'S_ptr': getitem_212, 'M_ptr': getitem_213, 'Y_ptr': empty_657, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_211 = getitem_212 = getitem_213 = triton_kernel_wrapper_mutation_81 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_2745: "bf16[512, 128, 784]" = torch.ops.aten.reshape.default(mul_405, [512, 128, 784]);  mul_405 = None
        view_2746: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(empty_657, [512, 196, 512]);  empty_657 = None
        view_2747: "bf16[512, 128, 784]" = torch.ops.aten.reshape.default(view_2746, [512, 128, 784]);  view_2746 = None
        
        # No stacktrace found for following nodes
        as_strided_default_76: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_38: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_76);  as_strided_default_76 = None
        as_strided_default_77: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_38, [128], [1], 0);  clone_default_38 = None
        as_strided_default_78: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_39: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_78);  as_strided_default_78 = None
        as_strided_default_79: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_39, [128], [1], 0);  clone_default_39 = None
        triton_kernel_wrapper_mutation_80 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 581, constant_args_idx = 803, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2747, 'DY': view_2745, 'DBETA': as_strided_default_77, 'DGAMMA': as_strided_default_79, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_80 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_658: "bf16[512, 128, 784]" = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_193: "bf16[512, 128, 784]" = torch.ops.aten.permute.default(empty_658, [0, 1, 2]);  empty_658 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_79 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 582, constant_args_idx = 804, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2747, 'DY': view_2745, 'INVSTD': rsqrt_16, 'GAMMA': primals_100, 'DBETA': as_strided_default_77, 'DGAMMA': as_strided_default_79, 'DX': permute_193, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024});  view_2747 = view_2745 = rsqrt_16 = primals_100 = triton_kernel_wrapper_mutation_79 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_659: "bf16[100352, 512]" = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_78 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 583, constant_args_idx = 805, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_204, 'S_ptr': getitem_205, 'M_ptr': getitem_206, 'Y_ptr': empty_659, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_204 = getitem_205 = getitem_206 = triton_kernel_wrapper_mutation_78 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_2763: "bf16[512, 128, 28, 28]" = torch.ops.aten.reshape.default(permute_193, [512, 128, 28, 28]);  permute_193 = None
        empty_660: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_73: "bf16[512, 128, 28, 28]" = torch.ops.aten.expand.default(empty_660, [512, 128, 28, 28]);  empty_660 = None
        convolution_backward_72 = torch.ops.aten.convolution_backward.default(view_2763, expand_73, convert_element_type_17, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_73 = convert_element_type_17 = None
        getitem_1119: "bf16[512, 128, 28, 28]" = convolution_backward_72[0];  convolution_backward_72 = None
        empty_661: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_74: "bf16[128, 128, 3, 3]" = torch.ops.aten.expand.default(empty_661, [128, 128, 3, 3]);  empty_661 = None
        view_2764: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(empty_659, [512, 196, 512]);  empty_659 = None
        view_2765: "bf16[512, 128, 28, 28]" = torch.ops.aten.reshape.default(view_2764, [512, 128, 28, 28]);  view_2764 = None
        convolution_backward_73 = torch.ops.aten.convolution_backward.default(view_2763, view_2765, expand_74, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_2763 = view_2765 = expand_74 = None
        getitem_1123: "bf16[128, 128, 3, 3]" = convolution_backward_73[1];  convolution_backward_73 = None
        convert_element_type_247: "f32[128, 128, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_1123, torch.float32);  getitem_1123 = None
        
        # No stacktrace found for following nodes
        as_strided_default_74: "i8[51380224]" = torch.ops.aten.as_strided.default(full_default_310, [51380224], [1], 0)
        clone_default_37: "i8[51380224]" = torch.ops.aten.clone.default(as_strided_default_74);  as_strided_default_74 = None
        as_strided_default_75: "i8[100352, 512]" = torch.ops.aten.as_strided.default(clone_default_37, [100352, 512], [512, 1], 0);  clone_default_37 = None
        triton_kernel_wrapper_mutation_77 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 806, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_203, 'Y_ptr': as_strided_default_75, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_203 = triton_kernel_wrapper_mutation_77 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_2768: "i8[512, 196, 512]" = torch.ops.aten.reshape.default(as_strided_default_75, [512, 196, 512]);  as_strided_default_75 = None
        view_2769: "i8[512, 128, 28, 28]" = torch.ops.aten.reshape.default(view_2768, [512, 128, 28, 28]);  view_2768 = None
        mul_406: "bf16[512, 128, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_1119, view_2769);  getitem_1119 = view_2769 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_662: "bf16[100352, 512]" = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_76 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 584, constant_args_idx = 807, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_198, 'S_ptr': getitem_199, 'M_ptr': getitem_200, 'Y_ptr': empty_662, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_198 = getitem_199 = getitem_200 = triton_kernel_wrapper_mutation_76 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_2784: "bf16[512, 128, 784]" = torch.ops.aten.reshape.default(mul_406, [512, 128, 784]);  mul_406 = None
        view_2785: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(empty_662, [512, 196, 512]);  empty_662 = None
        view_2786: "bf16[512, 128, 784]" = torch.ops.aten.reshape.default(view_2785, [512, 128, 784]);  view_2785 = None
        
        # No stacktrace found for following nodes
        as_strided_default_70: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_35: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_70);  as_strided_default_70 = None
        as_strided_default_71: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_35, [128], [1], 0);  clone_default_35 = None
        as_strided_default_72: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_36: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_72);  as_strided_default_72 = None
        as_strided_default_73: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_36, [128], [1], 0);  clone_default_36 = None
        triton_kernel_wrapper_mutation_75 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 585, constant_args_idx = 808, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2786, 'DY': view_2784, 'DBETA': as_strided_default_71, 'DGAMMA': as_strided_default_73, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_75 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_663: "bf16[512, 128, 784]" = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_194: "bf16[512, 128, 784]" = torch.ops.aten.permute.default(empty_663, [0, 1, 2]);  empty_663 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_74 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 586, constant_args_idx = 809, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2786, 'DY': view_2784, 'INVSTD': rsqrt_15, 'GAMMA': primals_94, 'DBETA': as_strided_default_71, 'DGAMMA': as_strided_default_73, 'DX': permute_194, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024});  view_2786 = view_2784 = rsqrt_15 = primals_94 = triton_kernel_wrapper_mutation_74 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_664: "bf16[401408, 512]" = torch.ops.aten.empty.memory_format([401408, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_73 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 587, constant_args_idx = 810, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_191, 'S_ptr': getitem_192, 'M_ptr': getitem_193, 'Y_ptr': empty_664, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_191 = getitem_192 = getitem_193 = triton_kernel_wrapper_mutation_73 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_2802: "bf16[512, 128, 28, 28]" = torch.ops.aten.reshape.default(permute_194, [512, 128, 28, 28]);  permute_194 = None
        empty_665: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_75: "bf16[512, 512, 28, 28]" = torch.ops.aten.expand.default(empty_665, [512, 512, 28, 28]);  empty_665 = None
        convolution_backward_74 = torch.ops.aten.convolution_backward.default(view_2802, expand_75, convert_element_type_16, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_75 = convert_element_type_16 = None
        getitem_1131: "bf16[512, 512, 28, 28]" = convolution_backward_74[0];  convolution_backward_74 = None
        empty_666: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_76: "bf16[128, 512, 1, 1]" = torch.ops.aten.expand.default(empty_666, [128, 512, 1, 1]);  empty_666 = None
        view_2803: "bf16[512, 784, 512]" = torch.ops.aten.reshape.default(empty_664, [512, 784, 512]);  empty_664 = None
        view_2804: "bf16[512, 512, 28, 28]" = torch.ops.aten.reshape.default(view_2803, [512, 512, 28, 28]);  view_2803 = None
        convolution_backward_75 = torch.ops.aten.convolution_backward.default(view_2802, view_2804, expand_76, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2802 = view_2804 = expand_76 = None
        getitem_1135: "bf16[128, 512, 1, 1]" = convolution_backward_75[1];  convolution_backward_75 = None
        convert_element_type_252: "f32[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1135, torch.float32);  getitem_1135 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_239: "bf16[512, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_404, getitem_1131);  mul_404 = getitem_1131 = None
        
        # No stacktrace found for following nodes
        as_strided_default_68: "i8[205520896]" = torch.ops.aten.as_strided.default(full_default_395, [205520896], [1], 0)
        clone_default_34: "i8[205520896]" = torch.ops.aten.clone.default(as_strided_default_68);  as_strided_default_68 = None
        as_strided_default_69: "i8[401408, 512]" = torch.ops.aten.as_strided.default(clone_default_34, [401408, 512], [512, 1], 0);  clone_default_34 = None
        triton_kernel_wrapper_mutation_72 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 811, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_190, 'Y_ptr': as_strided_default_69, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_190 = triton_kernel_wrapper_mutation_72 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_2807: "i8[512, 784, 512]" = torch.ops.aten.reshape.default(as_strided_default_69, [512, 784, 512]);  as_strided_default_69 = None
        view_2808: "i8[512, 512, 28, 28]" = torch.ops.aten.reshape.default(view_2807, [512, 512, 28, 28]);  view_2807 = None
        mul_407: "bf16[512, 512, 28, 28]" = torch.ops.aten.mul.Tensor(add_239, view_2808);  add_239 = view_2808 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_667: "bf16[401408, 512]" = torch.ops.aten.empty.memory_format([401408, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_71 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 588, constant_args_idx = 812, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_185, 'S_ptr': getitem_186, 'M_ptr': getitem_187, 'Y_ptr': empty_667, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_185 = getitem_186 = getitem_187 = triton_kernel_wrapper_mutation_71 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_2823: "bf16[512, 512, 784]" = torch.ops.aten.reshape.default(mul_407, [512, 512, 784]);  mul_407 = None
        view_2824: "bf16[512, 784, 512]" = torch.ops.aten.reshape.default(empty_667, [512, 784, 512]);  empty_667 = None
        view_2825: "bf16[512, 512, 784]" = torch.ops.aten.reshape.default(view_2824, [512, 512, 784]);  view_2824 = None
        
        # No stacktrace found for following nodes
        as_strided_default_64: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_32: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_64);  as_strided_default_64 = None
        as_strided_default_65: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_32, [512], [1], 0);  clone_default_32 = None
        as_strided_default_66: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_33: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_66);  as_strided_default_66 = None
        as_strided_default_67: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_33, [512], [1], 0);  clone_default_33 = None
        triton_kernel_wrapper_mutation_70 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 589, constant_args_idx = 813, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2825, 'DY': view_2823, 'DBETA': as_strided_default_65, 'DGAMMA': as_strided_default_67, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_70 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_668: "bf16[512, 512, 784]" = torch.ops.aten.empty.memory_format([512, 512, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_195: "bf16[512, 512, 784]" = torch.ops.aten.permute.default(empty_668, [0, 1, 2]);  empty_668 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_69 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 590, constant_args_idx = 814, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2825, 'DY': view_2823, 'INVSTD': rsqrt_14, 'GAMMA': primals_88, 'DBETA': as_strided_default_65, 'DGAMMA': as_strided_default_67, 'DX': permute_195, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024});  view_2825 = rsqrt_14 = primals_88 = triton_kernel_wrapper_mutation_69 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_669: "bf16[802816, 512]" = torch.ops.aten.empty.memory_format([802816, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_68 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 591, constant_args_idx = 815, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_178, 'S_ptr': getitem_179, 'M_ptr': getitem_180, 'Y_ptr': empty_669, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_178 = getitem_179 = getitem_180 = triton_kernel_wrapper_mutation_68 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_2841: "bf16[512, 512, 28, 28]" = torch.ops.aten.reshape.default(permute_195, [512, 512, 28, 28]);  permute_195 = None
        empty_670: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_77: "bf16[512, 256, 56, 56]" = torch.ops.aten.expand.default(empty_670, [512, 256, 56, 56]);  empty_670 = None
        convolution_backward_76 = torch.ops.aten.convolution_backward.default(view_2841, expand_77, convert_element_type_15, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_77 = convert_element_type_15 = None
        getitem_1143: "bf16[512, 256, 56, 56]" = convolution_backward_76[0];  convolution_backward_76 = None
        empty_671: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_78: "bf16[512, 256, 1, 1]" = torch.ops.aten.expand.default(empty_671, [512, 256, 1, 1]);  empty_671 = None
        view_2842: "bf16[512, 1568, 512]" = torch.ops.aten.reshape.default(empty_669, [512, 1568, 512]);  empty_669 = None
        view_2843: "bf16[512, 256, 56, 56]" = torch.ops.aten.reshape.default(view_2842, [512, 256, 56, 56]);  view_2842 = None
        convolution_backward_77 = torch.ops.aten.convolution_backward.default(view_2841, view_2843, expand_78, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2841 = view_2843 = expand_78 = None
        getitem_1147: "bf16[512, 256, 1, 1]" = convolution_backward_77[1];  convolution_backward_77 = None
        convert_element_type_257: "f32[512, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1147, torch.float32);  getitem_1147 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_672: "bf16[401408, 512]" = torch.ops.aten.empty.memory_format([401408, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_67 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 592, constant_args_idx = 816, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_175, 'S_ptr': getitem_176, 'M_ptr': getitem_177, 'Y_ptr': empty_672, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_175 = getitem_176 = getitem_177 = triton_kernel_wrapper_mutation_67 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_2859: "bf16[512, 784, 512]" = torch.ops.aten.reshape.default(empty_672, [512, 784, 512]);  empty_672 = None
        view_2860: "bf16[512, 512, 784]" = torch.ops.aten.reshape.default(view_2859, [512, 512, 784]);  view_2859 = None
        
        # No stacktrace found for following nodes
        as_strided_default_62: "f32[512]" = torch.ops.aten.as_strided.default(full_default_76, [512], [1], 0)
        clone_default_31: "f32[512]" = torch.ops.aten.clone.default(as_strided_default_62);  as_strided_default_62 = None
        as_strided_default_63: "f32[512]" = torch.ops.aten.as_strided.default(clone_default_31, [512], [1], 0);  clone_default_31 = None
        triton_kernel_wrapper_mutation_66 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 593, constant_args_idx = 817, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2860, 'DY': view_2823, 'DBETA': full_default_76, 'DGAMMA': as_strided_default_63, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_66 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_673: "bf16[512, 512, 784]" = torch.ops.aten.empty.memory_format([512, 512, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_196: "bf16[512, 512, 784]" = torch.ops.aten.permute.default(empty_673, [0, 1, 2]);  empty_673 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_65 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 594, constant_args_idx = 818, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2860, 'DY': view_2823, 'INVSTD': rsqrt_13, 'GAMMA': primals_82, 'DBETA': full_default_76, 'DGAMMA': as_strided_default_63, 'DX': permute_196, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024});  view_2860 = view_2823 = rsqrt_13 = primals_82 = triton_kernel_wrapper_mutation_65 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_674: "bf16[100352, 512]" = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_64 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 595, constant_args_idx = 819, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_168, 'S_ptr': getitem_169, 'M_ptr': getitem_170, 'Y_ptr': empty_674, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_168 = getitem_169 = getitem_170 = triton_kernel_wrapper_mutation_64 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_2876: "bf16[512, 512, 28, 28]" = torch.ops.aten.reshape.default(permute_196, [512, 512, 28, 28]);  permute_196 = None
        empty_675: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_79: "bf16[512, 128, 28, 28]" = torch.ops.aten.expand.default(empty_675, [512, 128, 28, 28]);  empty_675 = None
        convolution_backward_78 = torch.ops.aten.convolution_backward.default(view_2876, expand_79, convert_element_type_14, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_79 = convert_element_type_14 = None
        getitem_1154: "bf16[512, 128, 28, 28]" = convolution_backward_78[0];  convolution_backward_78 = None
        empty_676: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_80: "bf16[512, 128, 1, 1]" = torch.ops.aten.expand.default(empty_676, [512, 128, 1, 1]);  empty_676 = None
        view_2877: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(empty_674, [512, 196, 512]);  empty_674 = None
        view_2878: "bf16[512, 128, 28, 28]" = torch.ops.aten.reshape.default(view_2877, [512, 128, 28, 28]);  view_2877 = None
        convolution_backward_79 = torch.ops.aten.convolution_backward.default(view_2876, view_2878, expand_80, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2876 = view_2878 = expand_80 = None
        getitem_1158: "bf16[512, 128, 1, 1]" = convolution_backward_79[1];  convolution_backward_79 = None
        convert_element_type_262: "f32[512, 128, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1158, torch.float32);  getitem_1158 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_63 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 820, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_167, 'Y_ptr': full_default_310, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_167 = triton_kernel_wrapper_mutation_63 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_2881: "i8[512, 196, 512]" = torch.ops.aten.reshape.default(full_default_310, [512, 196, 512]);  full_default_310 = None
        view_2882: "i8[512, 128, 28, 28]" = torch.ops.aten.reshape.default(view_2881, [512, 128, 28, 28]);  view_2881 = None
        mul_408: "bf16[512, 128, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_1154, view_2882);  getitem_1154 = view_2882 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_677: "bf16[100352, 512]" = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_62 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 596, constant_args_idx = 821, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_162, 'S_ptr': getitem_163, 'M_ptr': getitem_164, 'Y_ptr': empty_677, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_162 = getitem_163 = getitem_164 = triton_kernel_wrapper_mutation_62 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_2897: "bf16[512, 128, 784]" = torch.ops.aten.reshape.default(mul_408, [512, 128, 784]);  mul_408 = None
        view_2898: "bf16[512, 196, 512]" = torch.ops.aten.reshape.default(empty_677, [512, 196, 512]);  empty_677 = None
        view_2899: "bf16[512, 128, 784]" = torch.ops.aten.reshape.default(view_2898, [512, 128, 784]);  view_2898 = None
        
        # No stacktrace found for following nodes
        as_strided_default_58: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_29: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_58);  as_strided_default_58 = None
        as_strided_default_59: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_29, [128], [1], 0);  clone_default_29 = None
        as_strided_default_60: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_30: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_60);  as_strided_default_60 = None
        as_strided_default_61: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_30, [128], [1], 0);  clone_default_30 = None
        triton_kernel_wrapper_mutation_61 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 597, constant_args_idx = 822, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2899, 'DY': view_2897, 'DBETA': as_strided_default_59, 'DGAMMA': as_strided_default_61, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_61 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_678: "bf16[512, 128, 784]" = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_197: "bf16[512, 128, 784]" = torch.ops.aten.permute.default(empty_678, [0, 1, 2]);  empty_678 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_60 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 598, constant_args_idx = 823, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2899, 'DY': view_2897, 'INVSTD': rsqrt_12, 'GAMMA': primals_76, 'DBETA': as_strided_default_59, 'DGAMMA': as_strided_default_61, 'DX': permute_197, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024});  view_2899 = view_2897 = rsqrt_12 = primals_76 = triton_kernel_wrapper_mutation_60 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_679: "bf16[401408, 512]" = torch.ops.aten.empty.memory_format([401408, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_59 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 599, constant_args_idx = 824, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_155, 'S_ptr': getitem_156, 'M_ptr': getitem_157, 'Y_ptr': empty_679, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_155 = getitem_156 = getitem_157 = triton_kernel_wrapper_mutation_59 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_2915: "bf16[512, 128, 28, 28]" = torch.ops.aten.reshape.default(permute_197, [512, 128, 28, 28]);  permute_197 = None
        empty_680: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_81: "bf16[512, 128, 56, 56]" = torch.ops.aten.expand.default(empty_680, [512, 128, 56, 56]);  empty_680 = None
        convolution_backward_80 = torch.ops.aten.convolution_backward.default(view_2915, expand_81, convert_element_type_13, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_81 = convert_element_type_13 = None
        getitem_1166: "bf16[512, 128, 56, 56]" = convolution_backward_80[0];  convolution_backward_80 = None
        empty_681: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_82: "bf16[128, 128, 3, 3]" = torch.ops.aten.expand.default(empty_681, [128, 128, 3, 3]);  empty_681 = None
        view_2916: "bf16[512, 784, 512]" = torch.ops.aten.reshape.default(empty_679, [512, 784, 512]);  empty_679 = None
        view_2917: "bf16[512, 128, 56, 56]" = torch.ops.aten.reshape.default(view_2916, [512, 128, 56, 56]);  view_2916 = None
        convolution_backward_81 = torch.ops.aten.convolution_backward.default(view_2915, view_2917, expand_82, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_2915 = view_2917 = expand_82 = None
        getitem_1170: "bf16[128, 128, 3, 3]" = convolution_backward_81[1];  convolution_backward_81 = None
        convert_element_type_267: "f32[128, 128, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_1170, torch.float32);  getitem_1170 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_58 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 825, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_154, 'Y_ptr': full_default_395, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_154 = triton_kernel_wrapper_mutation_58 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_2920: "i8[512, 784, 512]" = torch.ops.aten.reshape.default(full_default_395, [512, 784, 512]);  full_default_395 = None
        view_2921: "i8[512, 128, 56, 56]" = torch.ops.aten.reshape.default(view_2920, [512, 128, 56, 56]);  view_2920 = None
        mul_409: "bf16[512, 128, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_1166, view_2921);  getitem_1166 = view_2921 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_682: "bf16[401408, 512]" = torch.ops.aten.empty.memory_format([401408, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_57 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 600, constant_args_idx = 826, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_149, 'S_ptr': getitem_150, 'M_ptr': getitem_151, 'Y_ptr': empty_682, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_149 = getitem_150 = getitem_151 = triton_kernel_wrapper_mutation_57 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_2936: "bf16[512, 128, 3136]" = torch.ops.aten.reshape.default(mul_409, [512, 128, 3136]);  mul_409 = None
        view_2937: "bf16[512, 784, 512]" = torch.ops.aten.reshape.default(empty_682, [512, 784, 512]);  empty_682 = None
        view_2938: "bf16[512, 128, 3136]" = torch.ops.aten.reshape.default(view_2937, [512, 128, 3136]);  view_2937 = None
        
        # No stacktrace found for following nodes
        as_strided_default_56: "f32[128]" = torch.ops.aten.as_strided.default(full_default_64, [128], [1], 0)
        clone_default_28: "f32[128]" = torch.ops.aten.clone.default(as_strided_default_56);  as_strided_default_56 = None
        as_strided_default_57: "f32[128]" = torch.ops.aten.as_strided.default(clone_default_28, [128], [1], 0);  clone_default_28 = None
        triton_kernel_wrapper_mutation_56 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 601, constant_args_idx = 827, grid = [(128, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2938, 'DY': view_2936, 'DBETA': full_default_64, 'DGAMMA': as_strided_default_57, 'M': 1605632, 'HW': 3136, 'stride_n': 401408, 'stride_c': 3136, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_56 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_683: "bf16[512, 128, 3136]" = torch.ops.aten.empty.memory_format([512, 128, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_198: "bf16[512, 128, 3136]" = torch.ops.aten.permute.default(empty_683, [0, 1, 2]);  empty_683 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_55 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 602, constant_args_idx = 828, grid = [(128, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2938, 'DY': view_2936, 'INVSTD': rsqrt_11, 'GAMMA': primals_70, 'DBETA': full_default_64, 'DGAMMA': as_strided_default_57, 'DX': permute_198, 'M': 1605632, 'HW': 3136, 'stride_n': 401408, 'stride_c': 3136, 'BLOCK_M': 1024});  view_2938 = view_2936 = rsqrt_11 = primals_70 = triton_kernel_wrapper_mutation_55 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_684: "bf16[802816, 512]" = torch.ops.aten.empty.memory_format([802816, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_54 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 603, constant_args_idx = 829, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_142, 'S_ptr': getitem_143, 'M_ptr': getitem_144, 'Y_ptr': empty_684, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_142 = getitem_143 = getitem_144 = triton_kernel_wrapper_mutation_54 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_2954: "bf16[512, 128, 56, 56]" = torch.ops.aten.reshape.default(permute_198, [512, 128, 56, 56]);  permute_198 = None
        empty_685: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_83: "bf16[512, 256, 56, 56]" = torch.ops.aten.expand.default(empty_685, [512, 256, 56, 56]);  empty_685 = None
        convolution_backward_82 = torch.ops.aten.convolution_backward.default(view_2954, expand_83, convert_element_type_12, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_83 = convert_element_type_12 = None
        getitem_1178: "bf16[512, 256, 56, 56]" = convolution_backward_82[0];  convolution_backward_82 = None
        empty_686: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_84: "bf16[128, 256, 1, 1]" = torch.ops.aten.expand.default(empty_686, [128, 256, 1, 1]);  empty_686 = None
        view_2955: "bf16[512, 1568, 512]" = torch.ops.aten.reshape.default(empty_684, [512, 1568, 512]);  empty_684 = None
        view_2956: "bf16[512, 256, 56, 56]" = torch.ops.aten.reshape.default(view_2955, [512, 256, 56, 56]);  view_2955 = None
        convolution_backward_83 = torch.ops.aten.convolution_backward.default(view_2954, view_2956, expand_84, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2954 = view_2956 = expand_84 = None
        getitem_1182: "bf16[128, 256, 1, 1]" = convolution_backward_83[1];  convolution_backward_83 = None
        convert_element_type_272: "f32[128, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1182, torch.float32);  getitem_1182 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_240: "bf16[512, 256, 56, 56]" = torch.ops.aten.add.Tensor(getitem_1143, getitem_1178);  getitem_1143 = getitem_1178 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_433: "i8[802816, 512]" = torch.ops.aten.full.default([802816, 512], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        as_strided_default_54: "i8[411041792]" = torch.ops.aten.as_strided.default(full_default_433, [411041792], [1], 0)
        clone_default_27: "i8[411041792]" = torch.ops.aten.clone.default(as_strided_default_54);  as_strided_default_54 = None
        as_strided_default_55: "i8[802816, 512]" = torch.ops.aten.as_strided.default(clone_default_27, [802816, 512], [512, 1], 0);  clone_default_27 = None
        triton_kernel_wrapper_mutation_53 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 830, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_141, 'Y_ptr': as_strided_default_55, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_141 = triton_kernel_wrapper_mutation_53 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_2959: "i8[512, 1568, 512]" = torch.ops.aten.reshape.default(as_strided_default_55, [512, 1568, 512]);  as_strided_default_55 = None
        view_2960: "i8[512, 256, 56, 56]" = torch.ops.aten.reshape.default(view_2959, [512, 256, 56, 56]);  view_2959 = None
        mul_410: "bf16[512, 256, 56, 56]" = torch.ops.aten.mul.Tensor(add_240, view_2960);  add_240 = view_2960 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_687: "bf16[802816, 512]" = torch.ops.aten.empty.memory_format([802816, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_52 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 604, constant_args_idx = 831, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_136, 'S_ptr': getitem_137, 'M_ptr': getitem_138, 'Y_ptr': empty_687, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_136 = getitem_137 = getitem_138 = triton_kernel_wrapper_mutation_52 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_2975: "bf16[512, 256, 3136]" = torch.ops.aten.reshape.default(mul_410, [512, 256, 3136])
        view_2976: "bf16[512, 1568, 512]" = torch.ops.aten.reshape.default(empty_687, [512, 1568, 512]);  empty_687 = None
        view_2977: "bf16[512, 256, 3136]" = torch.ops.aten.reshape.default(view_2976, [512, 256, 3136]);  view_2976 = None
        
        # No stacktrace found for following nodes
        as_strided_default_50: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_25: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_50);  as_strided_default_50 = None
        as_strided_default_51: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_25, [256], [1], 0);  clone_default_25 = None
        as_strided_default_52: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_26: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_52);  as_strided_default_52 = None
        as_strided_default_53: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_26, [256], [1], 0);  clone_default_26 = None
        triton_kernel_wrapper_mutation_51 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 605, constant_args_idx = 832, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2977, 'DY': view_2975, 'DBETA': as_strided_default_51, 'DGAMMA': as_strided_default_53, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_51 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_688: "bf16[512, 256, 3136]" = torch.ops.aten.empty.memory_format([512, 256, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_199: "bf16[512, 256, 3136]" = torch.ops.aten.permute.default(empty_688, [0, 1, 2]);  empty_688 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_50 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 606, constant_args_idx = 833, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2977, 'DY': view_2975, 'INVSTD': rsqrt_10, 'GAMMA': primals_64, 'DBETA': as_strided_default_51, 'DGAMMA': as_strided_default_53, 'DX': permute_199, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024});  view_2977 = view_2975 = rsqrt_10 = primals_64 = triton_kernel_wrapper_mutation_50 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_689: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_49 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 607, constant_args_idx = 834, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_129, 'S_ptr': getitem_130, 'M_ptr': getitem_131, 'Y_ptr': empty_689, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_129 = getitem_130 = getitem_131 = triton_kernel_wrapper_mutation_49 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_2993: "bf16[512, 256, 56, 56]" = torch.ops.aten.reshape.default(permute_199, [512, 256, 56, 56]);  permute_199 = None
        empty_690: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_85: "bf16[512, 64, 56, 56]" = torch.ops.aten.expand.default(empty_690, [512, 64, 56, 56]);  empty_690 = None
        convolution_backward_84 = torch.ops.aten.convolution_backward.default(view_2993, expand_85, convert_element_type_11, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_85 = convert_element_type_11 = None
        getitem_1190: "bf16[512, 64, 56, 56]" = convolution_backward_84[0];  convolution_backward_84 = None
        empty_691: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_86: "bf16[256, 64, 1, 1]" = torch.ops.aten.expand.default(empty_691, [256, 64, 1, 1]);  empty_691 = None
        view_2994: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_689, [512, 392, 512]);  empty_689 = None
        view_2995: "bf16[512, 64, 56, 56]" = torch.ops.aten.reshape.default(view_2994, [512, 64, 56, 56]);  view_2994 = None
        convolution_backward_85 = torch.ops.aten.convolution_backward.default(view_2993, view_2995, expand_86, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2993 = view_2995 = expand_86 = None
        getitem_1194: "bf16[256, 64, 1, 1]" = convolution_backward_85[1];  convolution_backward_85 = None
        convert_element_type_277: "f32[256, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1194, torch.float32);  getitem_1194 = None
        
        # No stacktrace found for following nodes
        as_strided_default_48: "i8[102760448]" = torch.ops.aten.as_strided.default(full_default_339, [102760448], [1], 0)
        clone_default_24: "i8[102760448]" = torch.ops.aten.clone.default(as_strided_default_48);  as_strided_default_48 = None
        as_strided_default_49: "i8[200704, 512]" = torch.ops.aten.as_strided.default(clone_default_24, [200704, 512], [512, 1], 0);  clone_default_24 = None
        triton_kernel_wrapper_mutation_48 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 835, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_128, 'Y_ptr': as_strided_default_49, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_128 = triton_kernel_wrapper_mutation_48 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_2998: "i8[512, 392, 512]" = torch.ops.aten.reshape.default(as_strided_default_49, [512, 392, 512]);  as_strided_default_49 = None
        view_2999: "i8[512, 64, 56, 56]" = torch.ops.aten.reshape.default(view_2998, [512, 64, 56, 56]);  view_2998 = None
        mul_411: "bf16[512, 64, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_1190, view_2999);  getitem_1190 = view_2999 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_692: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_47 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 608, constant_args_idx = 836, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_123, 'S_ptr': getitem_124, 'M_ptr': getitem_125, 'Y_ptr': empty_692, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_123 = getitem_124 = getitem_125 = triton_kernel_wrapper_mutation_47 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_3014: "bf16[512, 64, 3136]" = torch.ops.aten.reshape.default(mul_411, [512, 64, 3136]);  mul_411 = None
        view_3015: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_692, [512, 392, 512]);  empty_692 = None
        view_3016: "bf16[512, 64, 3136]" = torch.ops.aten.reshape.default(view_3015, [512, 64, 3136]);  view_3015 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        full_default: "f32[64]" = torch.ops.aten.full.default([64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        as_strided_default_44: "f32[64]" = torch.ops.aten.as_strided.default(full_default, [64], [1], 0)
        clone_default_22: "f32[64]" = torch.ops.aten.clone.default(as_strided_default_44);  as_strided_default_44 = None
        as_strided_default_45: "f32[64]" = torch.ops.aten.as_strided.default(clone_default_22, [64], [1], 0);  clone_default_22 = None
        as_strided_default_46: "f32[64]" = torch.ops.aten.as_strided.default(full_default, [64], [1], 0)
        clone_default_23: "f32[64]" = torch.ops.aten.clone.default(as_strided_default_46);  as_strided_default_46 = None
        as_strided_default_47: "f32[64]" = torch.ops.aten.as_strided.default(clone_default_23, [64], [1], 0);  clone_default_23 = None
        triton_kernel_wrapper_mutation_46 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 609, constant_args_idx = 837, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3016, 'DY': view_3014, 'DBETA': as_strided_default_45, 'DGAMMA': as_strided_default_47, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_46 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_693: "bf16[512, 64, 3136]" = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_200: "bf16[512, 64, 3136]" = torch.ops.aten.permute.default(empty_693, [0, 1, 2]);  empty_693 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_45 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 610, constant_args_idx = 838, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3016, 'DY': view_3014, 'INVSTD': rsqrt_9, 'GAMMA': primals_58, 'DBETA': as_strided_default_45, 'DGAMMA': as_strided_default_47, 'DX': permute_200, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024});  view_3016 = view_3014 = rsqrt_9 = primals_58 = triton_kernel_wrapper_mutation_45 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_694: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_44 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 611, constant_args_idx = 839, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_116, 'S_ptr': getitem_117, 'M_ptr': getitem_118, 'Y_ptr': empty_694, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_116 = getitem_117 = getitem_118 = triton_kernel_wrapper_mutation_44 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_3032: "bf16[512, 64, 56, 56]" = torch.ops.aten.reshape.default(permute_200, [512, 64, 56, 56]);  permute_200 = None
        empty_695: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_87: "bf16[512, 64, 56, 56]" = torch.ops.aten.expand.default(empty_695, [512, 64, 56, 56]);  empty_695 = None
        convolution_backward_86 = torch.ops.aten.convolution_backward.default(view_3032, expand_87, convert_element_type_10, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_87 = convert_element_type_10 = None
        getitem_1202: "bf16[512, 64, 56, 56]" = convolution_backward_86[0];  convolution_backward_86 = None
        empty_696: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_88: "bf16[64, 64, 3, 3]" = torch.ops.aten.expand.default(empty_696, [64, 64, 3, 3]);  empty_696 = None
        view_3033: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_694, [512, 392, 512]);  empty_694 = None
        view_3034: "bf16[512, 64, 56, 56]" = torch.ops.aten.reshape.default(view_3033, [512, 64, 56, 56]);  view_3033 = None
        convolution_backward_87 = torch.ops.aten.convolution_backward.default(view_3032, view_3034, expand_88, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_3032 = view_3034 = expand_88 = None
        getitem_1206: "bf16[64, 64, 3, 3]" = convolution_backward_87[1];  convolution_backward_87 = None
        convert_element_type_282: "f32[64, 64, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_1206, torch.float32);  getitem_1206 = None
        
        # No stacktrace found for following nodes
        as_strided_default_42: "i8[102760448]" = torch.ops.aten.as_strided.default(full_default_339, [102760448], [1], 0)
        clone_default_21: "i8[102760448]" = torch.ops.aten.clone.default(as_strided_default_42);  as_strided_default_42 = None
        as_strided_default_43: "i8[200704, 512]" = torch.ops.aten.as_strided.default(clone_default_21, [200704, 512], [512, 1], 0);  clone_default_21 = None
        triton_kernel_wrapper_mutation_43 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 840, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_115, 'Y_ptr': as_strided_default_43, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_115 = triton_kernel_wrapper_mutation_43 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_3037: "i8[512, 392, 512]" = torch.ops.aten.reshape.default(as_strided_default_43, [512, 392, 512]);  as_strided_default_43 = None
        view_3038: "i8[512, 64, 56, 56]" = torch.ops.aten.reshape.default(view_3037, [512, 64, 56, 56]);  view_3037 = None
        mul_412: "bf16[512, 64, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_1202, view_3038);  getitem_1202 = view_3038 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_697: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_42 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 612, constant_args_idx = 841, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_110, 'S_ptr': getitem_111, 'M_ptr': getitem_112, 'Y_ptr': empty_697, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_110 = getitem_111 = getitem_112 = triton_kernel_wrapper_mutation_42 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_3053: "bf16[512, 64, 3136]" = torch.ops.aten.reshape.default(mul_412, [512, 64, 3136]);  mul_412 = None
        view_3054: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_697, [512, 392, 512]);  empty_697 = None
        view_3055: "bf16[512, 64, 3136]" = torch.ops.aten.reshape.default(view_3054, [512, 64, 3136]);  view_3054 = None
        
        # No stacktrace found for following nodes
        as_strided_default_38: "f32[64]" = torch.ops.aten.as_strided.default(full_default, [64], [1], 0)
        clone_default_19: "f32[64]" = torch.ops.aten.clone.default(as_strided_default_38);  as_strided_default_38 = None
        as_strided_default_39: "f32[64]" = torch.ops.aten.as_strided.default(clone_default_19, [64], [1], 0);  clone_default_19 = None
        as_strided_default_40: "f32[64]" = torch.ops.aten.as_strided.default(full_default, [64], [1], 0)
        clone_default_20: "f32[64]" = torch.ops.aten.clone.default(as_strided_default_40);  as_strided_default_40 = None
        as_strided_default_41: "f32[64]" = torch.ops.aten.as_strided.default(clone_default_20, [64], [1], 0);  clone_default_20 = None
        triton_kernel_wrapper_mutation_41 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 613, constant_args_idx = 842, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3055, 'DY': view_3053, 'DBETA': as_strided_default_39, 'DGAMMA': as_strided_default_41, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_41 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_698: "bf16[512, 64, 3136]" = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_201: "bf16[512, 64, 3136]" = torch.ops.aten.permute.default(empty_698, [0, 1, 2]);  empty_698 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_40 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 614, constant_args_idx = 843, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3055, 'DY': view_3053, 'INVSTD': rsqrt_8, 'GAMMA': primals_52, 'DBETA': as_strided_default_39, 'DGAMMA': as_strided_default_41, 'DX': permute_201, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024});  view_3055 = view_3053 = rsqrt_8 = primals_52 = triton_kernel_wrapper_mutation_40 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_699: "bf16[802816, 512]" = torch.ops.aten.empty.memory_format([802816, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_39 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 615, constant_args_idx = 844, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_103, 'S_ptr': getitem_104, 'M_ptr': getitem_105, 'Y_ptr': empty_699, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_103 = getitem_104 = getitem_105 = triton_kernel_wrapper_mutation_39 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_3071: "bf16[512, 64, 56, 56]" = torch.ops.aten.reshape.default(permute_201, [512, 64, 56, 56]);  permute_201 = None
        empty_700: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_89: "bf16[512, 256, 56, 56]" = torch.ops.aten.expand.default(empty_700, [512, 256, 56, 56]);  empty_700 = None
        convolution_backward_88 = torch.ops.aten.convolution_backward.default(view_3071, expand_89, convert_element_type_9, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_89 = convert_element_type_9 = None
        getitem_1214: "bf16[512, 256, 56, 56]" = convolution_backward_88[0];  convolution_backward_88 = None
        empty_701: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_90: "bf16[64, 256, 1, 1]" = torch.ops.aten.expand.default(empty_701, [64, 256, 1, 1]);  empty_701 = None
        view_3072: "bf16[512, 1568, 512]" = torch.ops.aten.reshape.default(empty_699, [512, 1568, 512]);  empty_699 = None
        view_3073: "bf16[512, 256, 56, 56]" = torch.ops.aten.reshape.default(view_3072, [512, 256, 56, 56]);  view_3072 = None
        convolution_backward_89 = torch.ops.aten.convolution_backward.default(view_3071, view_3073, expand_90, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_3071 = view_3073 = expand_90 = None
        getitem_1218: "bf16[64, 256, 1, 1]" = convolution_backward_89[1];  convolution_backward_89 = None
        convert_element_type_287: "f32[64, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1218, torch.float32);  getitem_1218 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_241: "bf16[512, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_410, getitem_1214);  mul_410 = getitem_1214 = None
        
        # No stacktrace found for following nodes
        as_strided_default_36: "i8[411041792]" = torch.ops.aten.as_strided.default(full_default_433, [411041792], [1], 0)
        clone_default_18: "i8[411041792]" = torch.ops.aten.clone.default(as_strided_default_36);  as_strided_default_36 = None
        as_strided_default_37: "i8[802816, 512]" = torch.ops.aten.as_strided.default(clone_default_18, [802816, 512], [512, 1], 0);  clone_default_18 = None
        triton_kernel_wrapper_mutation_38 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 845, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_102, 'Y_ptr': as_strided_default_37, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_102 = triton_kernel_wrapper_mutation_38 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_3076: "i8[512, 1568, 512]" = torch.ops.aten.reshape.default(as_strided_default_37, [512, 1568, 512]);  as_strided_default_37 = None
        view_3077: "i8[512, 256, 56, 56]" = torch.ops.aten.reshape.default(view_3076, [512, 256, 56, 56]);  view_3076 = None
        mul_413: "bf16[512, 256, 56, 56]" = torch.ops.aten.mul.Tensor(add_241, view_3077);  add_241 = view_3077 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_702: "bf16[802816, 512]" = torch.ops.aten.empty.memory_format([802816, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_37 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 616, constant_args_idx = 846, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_97, 'S_ptr': getitem_98, 'M_ptr': getitem_99, 'Y_ptr': empty_702, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_97 = getitem_98 = getitem_99 = triton_kernel_wrapper_mutation_37 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_3092: "bf16[512, 256, 3136]" = torch.ops.aten.reshape.default(mul_413, [512, 256, 3136])
        view_3093: "bf16[512, 1568, 512]" = torch.ops.aten.reshape.default(empty_702, [512, 1568, 512]);  empty_702 = None
        view_3094: "bf16[512, 256, 3136]" = torch.ops.aten.reshape.default(view_3093, [512, 256, 3136]);  view_3093 = None
        
        # No stacktrace found for following nodes
        as_strided_default_32: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_16: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_32);  as_strided_default_32 = None
        as_strided_default_33: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_16, [256], [1], 0);  clone_default_16 = None
        as_strided_default_34: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_17: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_34);  as_strided_default_34 = None
        as_strided_default_35: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_17, [256], [1], 0);  clone_default_17 = None
        triton_kernel_wrapper_mutation_36 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 617, constant_args_idx = 847, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3094, 'DY': view_3092, 'DBETA': as_strided_default_33, 'DGAMMA': as_strided_default_35, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_36 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_703: "bf16[512, 256, 3136]" = torch.ops.aten.empty.memory_format([512, 256, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_202: "bf16[512, 256, 3136]" = torch.ops.aten.permute.default(empty_703, [0, 1, 2]);  empty_703 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_35 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 618, constant_args_idx = 848, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3094, 'DY': view_3092, 'INVSTD': rsqrt_7, 'GAMMA': primals_46, 'DBETA': as_strided_default_33, 'DGAMMA': as_strided_default_35, 'DX': permute_202, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024});  view_3094 = view_3092 = rsqrt_7 = primals_46 = triton_kernel_wrapper_mutation_35 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_704: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_34 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 619, constant_args_idx = 849, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_90, 'S_ptr': getitem_91, 'M_ptr': getitem_92, 'Y_ptr': empty_704, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_90 = getitem_91 = getitem_92 = triton_kernel_wrapper_mutation_34 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_3110: "bf16[512, 256, 56, 56]" = torch.ops.aten.reshape.default(permute_202, [512, 256, 56, 56]);  permute_202 = None
        empty_705: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_91: "bf16[512, 64, 56, 56]" = torch.ops.aten.expand.default(empty_705, [512, 64, 56, 56]);  empty_705 = None
        convolution_backward_90 = torch.ops.aten.convolution_backward.default(view_3110, expand_91, convert_element_type_8, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_91 = convert_element_type_8 = None
        getitem_1226: "bf16[512, 64, 56, 56]" = convolution_backward_90[0];  convolution_backward_90 = None
        empty_706: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_92: "bf16[256, 64, 1, 1]" = torch.ops.aten.expand.default(empty_706, [256, 64, 1, 1]);  empty_706 = None
        view_3111: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_704, [512, 392, 512]);  empty_704 = None
        view_3112: "bf16[512, 64, 56, 56]" = torch.ops.aten.reshape.default(view_3111, [512, 64, 56, 56]);  view_3111 = None
        convolution_backward_91 = torch.ops.aten.convolution_backward.default(view_3110, view_3112, expand_92, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_3110 = view_3112 = expand_92 = None
        getitem_1230: "bf16[256, 64, 1, 1]" = convolution_backward_91[1];  convolution_backward_91 = None
        convert_element_type_292: "f32[256, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1230, torch.float32);  getitem_1230 = None
        
        # No stacktrace found for following nodes
        as_strided_default_30: "i8[102760448]" = torch.ops.aten.as_strided.default(full_default_339, [102760448], [1], 0)
        clone_default_15: "i8[102760448]" = torch.ops.aten.clone.default(as_strided_default_30);  as_strided_default_30 = None
        as_strided_default_31: "i8[200704, 512]" = torch.ops.aten.as_strided.default(clone_default_15, [200704, 512], [512, 1], 0);  clone_default_15 = None
        triton_kernel_wrapper_mutation_33 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 850, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_89, 'Y_ptr': as_strided_default_31, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_89 = triton_kernel_wrapper_mutation_33 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_3115: "i8[512, 392, 512]" = torch.ops.aten.reshape.default(as_strided_default_31, [512, 392, 512]);  as_strided_default_31 = None
        view_3116: "i8[512, 64, 56, 56]" = torch.ops.aten.reshape.default(view_3115, [512, 64, 56, 56]);  view_3115 = None
        mul_414: "bf16[512, 64, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_1226, view_3116);  getitem_1226 = view_3116 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_707: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_32 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 620, constant_args_idx = 851, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_84, 'S_ptr': getitem_85, 'M_ptr': getitem_86, 'Y_ptr': empty_707, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_84 = getitem_85 = getitem_86 = triton_kernel_wrapper_mutation_32 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_3131: "bf16[512, 64, 3136]" = torch.ops.aten.reshape.default(mul_414, [512, 64, 3136]);  mul_414 = None
        view_3132: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_707, [512, 392, 512]);  empty_707 = None
        view_3133: "bf16[512, 64, 3136]" = torch.ops.aten.reshape.default(view_3132, [512, 64, 3136]);  view_3132 = None
        
        # No stacktrace found for following nodes
        as_strided_default_26: "f32[64]" = torch.ops.aten.as_strided.default(full_default, [64], [1], 0)
        clone_default_13: "f32[64]" = torch.ops.aten.clone.default(as_strided_default_26);  as_strided_default_26 = None
        as_strided_default_27: "f32[64]" = torch.ops.aten.as_strided.default(clone_default_13, [64], [1], 0);  clone_default_13 = None
        as_strided_default_28: "f32[64]" = torch.ops.aten.as_strided.default(full_default, [64], [1], 0)
        clone_default_14: "f32[64]" = torch.ops.aten.clone.default(as_strided_default_28);  as_strided_default_28 = None
        as_strided_default_29: "f32[64]" = torch.ops.aten.as_strided.default(clone_default_14, [64], [1], 0);  clone_default_14 = None
        triton_kernel_wrapper_mutation_31 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 621, constant_args_idx = 852, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3133, 'DY': view_3131, 'DBETA': as_strided_default_27, 'DGAMMA': as_strided_default_29, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_31 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_708: "bf16[512, 64, 3136]" = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_203: "bf16[512, 64, 3136]" = torch.ops.aten.permute.default(empty_708, [0, 1, 2]);  empty_708 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_30 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 622, constant_args_idx = 853, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3133, 'DY': view_3131, 'INVSTD': rsqrt_6, 'GAMMA': primals_40, 'DBETA': as_strided_default_27, 'DGAMMA': as_strided_default_29, 'DX': permute_203, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024});  view_3133 = view_3131 = rsqrt_6 = primals_40 = triton_kernel_wrapper_mutation_30 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_709: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_29 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 623, constant_args_idx = 854, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_77, 'S_ptr': getitem_78, 'M_ptr': getitem_79, 'Y_ptr': empty_709, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_77 = getitem_78 = getitem_79 = triton_kernel_wrapper_mutation_29 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_3149: "bf16[512, 64, 56, 56]" = torch.ops.aten.reshape.default(permute_203, [512, 64, 56, 56]);  permute_203 = None
        empty_710: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_93: "bf16[512, 64, 56, 56]" = torch.ops.aten.expand.default(empty_710, [512, 64, 56, 56]);  empty_710 = None
        convolution_backward_92 = torch.ops.aten.convolution_backward.default(view_3149, expand_93, convert_element_type_7, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_93 = convert_element_type_7 = None
        getitem_1238: "bf16[512, 64, 56, 56]" = convolution_backward_92[0];  convolution_backward_92 = None
        empty_711: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_94: "bf16[64, 64, 3, 3]" = torch.ops.aten.expand.default(empty_711, [64, 64, 3, 3]);  empty_711 = None
        view_3150: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_709, [512, 392, 512]);  empty_709 = None
        view_3151: "bf16[512, 64, 56, 56]" = torch.ops.aten.reshape.default(view_3150, [512, 64, 56, 56]);  view_3150 = None
        convolution_backward_93 = torch.ops.aten.convolution_backward.default(view_3149, view_3151, expand_94, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_3149 = view_3151 = expand_94 = None
        getitem_1242: "bf16[64, 64, 3, 3]" = convolution_backward_93[1];  convolution_backward_93 = None
        convert_element_type_297: "f32[64, 64, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_1242, torch.float32);  getitem_1242 = None
        
        # No stacktrace found for following nodes
        as_strided_default_24: "i8[102760448]" = torch.ops.aten.as_strided.default(full_default_339, [102760448], [1], 0)
        clone_default_12: "i8[102760448]" = torch.ops.aten.clone.default(as_strided_default_24);  as_strided_default_24 = None
        as_strided_default_25: "i8[200704, 512]" = torch.ops.aten.as_strided.default(clone_default_12, [200704, 512], [512, 1], 0);  clone_default_12 = None
        triton_kernel_wrapper_mutation_28 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 855, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_76, 'Y_ptr': as_strided_default_25, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_76 = triton_kernel_wrapper_mutation_28 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_3154: "i8[512, 392, 512]" = torch.ops.aten.reshape.default(as_strided_default_25, [512, 392, 512]);  as_strided_default_25 = None
        view_3155: "i8[512, 64, 56, 56]" = torch.ops.aten.reshape.default(view_3154, [512, 64, 56, 56]);  view_3154 = None
        mul_415: "bf16[512, 64, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_1238, view_3155);  getitem_1238 = view_3155 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_712: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_27 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 624, constant_args_idx = 856, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_71, 'S_ptr': getitem_72, 'M_ptr': getitem_73, 'Y_ptr': empty_712, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_71 = getitem_72 = getitem_73 = triton_kernel_wrapper_mutation_27 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_3170: "bf16[512, 64, 3136]" = torch.ops.aten.reshape.default(mul_415, [512, 64, 3136]);  mul_415 = None
        view_3171: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_712, [512, 392, 512]);  empty_712 = None
        view_3172: "bf16[512, 64, 3136]" = torch.ops.aten.reshape.default(view_3171, [512, 64, 3136]);  view_3171 = None
        
        # No stacktrace found for following nodes
        as_strided_default_20: "f32[64]" = torch.ops.aten.as_strided.default(full_default, [64], [1], 0)
        clone_default_10: "f32[64]" = torch.ops.aten.clone.default(as_strided_default_20);  as_strided_default_20 = None
        as_strided_default_21: "f32[64]" = torch.ops.aten.as_strided.default(clone_default_10, [64], [1], 0);  clone_default_10 = None
        as_strided_default_22: "f32[64]" = torch.ops.aten.as_strided.default(full_default, [64], [1], 0)
        clone_default_11: "f32[64]" = torch.ops.aten.clone.default(as_strided_default_22);  as_strided_default_22 = None
        as_strided_default_23: "f32[64]" = torch.ops.aten.as_strided.default(clone_default_11, [64], [1], 0);  clone_default_11 = None
        triton_kernel_wrapper_mutation_26 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 625, constant_args_idx = 857, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3172, 'DY': view_3170, 'DBETA': as_strided_default_21, 'DGAMMA': as_strided_default_23, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_26 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_713: "bf16[512, 64, 3136]" = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_204: "bf16[512, 64, 3136]" = torch.ops.aten.permute.default(empty_713, [0, 1, 2]);  empty_713 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_25 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 626, constant_args_idx = 858, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3172, 'DY': view_3170, 'INVSTD': rsqrt_5, 'GAMMA': primals_34, 'DBETA': as_strided_default_21, 'DGAMMA': as_strided_default_23, 'DX': permute_204, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024});  view_3172 = view_3170 = rsqrt_5 = primals_34 = triton_kernel_wrapper_mutation_25 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_714: "bf16[802816, 512]" = torch.ops.aten.empty.memory_format([802816, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_24 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 627, constant_args_idx = 859, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_64, 'S_ptr': getitem_65, 'M_ptr': getitem_66, 'Y_ptr': empty_714, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_64 = getitem_65 = getitem_66 = triton_kernel_wrapper_mutation_24 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_3188: "bf16[512, 64, 56, 56]" = torch.ops.aten.reshape.default(permute_204, [512, 64, 56, 56]);  permute_204 = None
        empty_715: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_95: "bf16[512, 256, 56, 56]" = torch.ops.aten.expand.default(empty_715, [512, 256, 56, 56]);  empty_715 = None
        convolution_backward_94 = torch.ops.aten.convolution_backward.default(view_3188, expand_95, convert_element_type_6, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_95 = convert_element_type_6 = None
        getitem_1250: "bf16[512, 256, 56, 56]" = convolution_backward_94[0];  convolution_backward_94 = None
        empty_716: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_96: "bf16[64, 256, 1, 1]" = torch.ops.aten.expand.default(empty_716, [64, 256, 1, 1]);  empty_716 = None
        view_3189: "bf16[512, 1568, 512]" = torch.ops.aten.reshape.default(empty_714, [512, 1568, 512]);  empty_714 = None
        view_3190: "bf16[512, 256, 56, 56]" = torch.ops.aten.reshape.default(view_3189, [512, 256, 56, 56]);  view_3189 = None
        convolution_backward_95 = torch.ops.aten.convolution_backward.default(view_3188, view_3190, expand_96, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_3188 = view_3190 = expand_96 = None
        getitem_1254: "bf16[64, 256, 1, 1]" = convolution_backward_95[1];  convolution_backward_95 = None
        convert_element_type_302: "f32[64, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1254, torch.float32);  getitem_1254 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_242: "bf16[512, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_413, getitem_1250);  mul_413 = getitem_1250 = None
        
        # No stacktrace found for following nodes
        as_strided_default_18: "i8[411041792]" = torch.ops.aten.as_strided.default(full_default_433, [411041792], [1], 0)
        clone_default_9: "i8[411041792]" = torch.ops.aten.clone.default(as_strided_default_18);  as_strided_default_18 = None
        as_strided_default_19: "i8[802816, 512]" = torch.ops.aten.as_strided.default(clone_default_9, [802816, 512], [512, 1], 0);  clone_default_9 = None
        triton_kernel_wrapper_mutation_23 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 860, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_63, 'Y_ptr': as_strided_default_19, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_63 = triton_kernel_wrapper_mutation_23 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_3193: "i8[512, 1568, 512]" = torch.ops.aten.reshape.default(as_strided_default_19, [512, 1568, 512]);  as_strided_default_19 = None
        view_3194: "i8[512, 256, 56, 56]" = torch.ops.aten.reshape.default(view_3193, [512, 256, 56, 56]);  view_3193 = None
        mul_416: "bf16[512, 256, 56, 56]" = torch.ops.aten.mul.Tensor(add_242, view_3194);  add_242 = view_3194 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_717: "bf16[802816, 512]" = torch.ops.aten.empty.memory_format([802816, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_22 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 628, constant_args_idx = 861, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_58, 'S_ptr': getitem_59, 'M_ptr': getitem_60, 'Y_ptr': empty_717, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_58 = getitem_59 = getitem_60 = triton_kernel_wrapper_mutation_22 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_3209: "bf16[512, 256, 3136]" = torch.ops.aten.reshape.default(mul_416, [512, 256, 3136]);  mul_416 = None
        view_3210: "bf16[512, 1568, 512]" = torch.ops.aten.reshape.default(empty_717, [512, 1568, 512]);  empty_717 = None
        view_3211: "bf16[512, 256, 3136]" = torch.ops.aten.reshape.default(view_3210, [512, 256, 3136]);  view_3210 = None
        
        # No stacktrace found for following nodes
        as_strided_default_14: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_7: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_14);  as_strided_default_14 = None
        as_strided_default_15: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_7, [256], [1], 0);  clone_default_7 = None
        as_strided_default_16: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_8: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_16);  as_strided_default_16 = None
        as_strided_default_17: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_8, [256], [1], 0);  clone_default_8 = None
        triton_kernel_wrapper_mutation_21 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 629, constant_args_idx = 862, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3211, 'DY': view_3209, 'DBETA': as_strided_default_15, 'DGAMMA': as_strided_default_17, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_21 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_718: "bf16[512, 256, 3136]" = torch.ops.aten.empty.memory_format([512, 256, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_205: "bf16[512, 256, 3136]" = torch.ops.aten.permute.default(empty_718, [0, 1, 2]);  empty_718 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_20 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 630, constant_args_idx = 863, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3211, 'DY': view_3209, 'INVSTD': rsqrt_4, 'GAMMA': primals_28, 'DBETA': as_strided_default_15, 'DGAMMA': as_strided_default_17, 'DX': permute_205, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024});  view_3211 = rsqrt_4 = primals_28 = triton_kernel_wrapper_mutation_20 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_719: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_19 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 631, constant_args_idx = 864, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_51, 'S_ptr': getitem_52, 'M_ptr': getitem_53, 'Y_ptr': empty_719, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_51 = getitem_52 = getitem_53 = triton_kernel_wrapper_mutation_19 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_3227: "bf16[512, 256, 56, 56]" = torch.ops.aten.reshape.default(permute_205, [512, 256, 56, 56]);  permute_205 = None
        empty_720: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_97: "bf16[512, 64, 56, 56]" = torch.ops.aten.expand.default(empty_720, [512, 64, 56, 56]);  empty_720 = None
        convolution_backward_96 = torch.ops.aten.convolution_backward.default(view_3227, expand_97, convert_element_type_5, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_97 = convert_element_type_5 = None
        getitem_1262: "bf16[512, 64, 56, 56]" = convolution_backward_96[0];  convolution_backward_96 = None
        empty_721: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_98: "bf16[256, 64, 1, 1]" = torch.ops.aten.expand.default(empty_721, [256, 64, 1, 1]);  empty_721 = None
        view_3228: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_719, [512, 392, 512]);  empty_719 = None
        view_3229: "bf16[512, 64, 56, 56]" = torch.ops.aten.reshape.default(view_3228, [512, 64, 56, 56]);  view_3228 = None
        convolution_backward_97 = torch.ops.aten.convolution_backward.default(view_3227, view_3229, expand_98, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_3227 = view_3229 = expand_98 = None
        getitem_1266: "bf16[256, 64, 1, 1]" = convolution_backward_97[1];  convolution_backward_97 = None
        convert_element_type_307: "f32[256, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1266, torch.float32);  getitem_1266 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_722: "bf16[802816, 512]" = torch.ops.aten.empty.memory_format([802816, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_18 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 632, constant_args_idx = 865, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_48, 'S_ptr': getitem_49, 'M_ptr': getitem_50, 'Y_ptr': empty_722, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_48 = getitem_49 = getitem_50 = triton_kernel_wrapper_mutation_18 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_3245: "bf16[512, 1568, 512]" = torch.ops.aten.reshape.default(empty_722, [512, 1568, 512]);  empty_722 = None
        view_3246: "bf16[512, 256, 3136]" = torch.ops.aten.reshape.default(view_3245, [512, 256, 3136]);  view_3245 = None
        
        # No stacktrace found for following nodes
        as_strided_default_12: "f32[256]" = torch.ops.aten.as_strided.default(full_default_18, [256], [1], 0)
        clone_default_6: "f32[256]" = torch.ops.aten.clone.default(as_strided_default_12);  as_strided_default_12 = None
        as_strided_default_13: "f32[256]" = torch.ops.aten.as_strided.default(clone_default_6, [256], [1], 0);  clone_default_6 = None
        triton_kernel_wrapper_mutation_17 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 633, constant_args_idx = 866, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3246, 'DY': view_3209, 'DBETA': full_default_18, 'DGAMMA': as_strided_default_13, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_17 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_723: "bf16[512, 256, 3136]" = torch.ops.aten.empty.memory_format([512, 256, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_206: "bf16[512, 256, 3136]" = torch.ops.aten.permute.default(empty_723, [0, 1, 2]);  empty_723 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_16 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 634, constant_args_idx = 867, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3246, 'DY': view_3209, 'INVSTD': rsqrt_3, 'GAMMA': primals_22, 'DBETA': full_default_18, 'DGAMMA': as_strided_default_13, 'DX': permute_206, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024});  view_3246 = view_3209 = rsqrt_3 = primals_22 = triton_kernel_wrapper_mutation_16 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_724: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_15 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 635, constant_args_idx = 868, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_41, 'S_ptr': getitem_42, 'M_ptr': getitem_43, 'Y_ptr': empty_724, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_41 = getitem_42 = getitem_43 = triton_kernel_wrapper_mutation_15 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_3262: "bf16[512, 256, 56, 56]" = torch.ops.aten.reshape.default(permute_206, [512, 256, 56, 56]);  permute_206 = None
        empty_725: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_99: "bf16[512, 64, 56, 56]" = torch.ops.aten.expand.default(empty_725, [512, 64, 56, 56]);  empty_725 = None
        convolution_backward_98 = torch.ops.aten.convolution_backward.default(view_3262, expand_99, convert_element_type_4, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_99 = convert_element_type_4 = None
        getitem_1273: "bf16[512, 64, 56, 56]" = convolution_backward_98[0];  convolution_backward_98 = None
        empty_726: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_100: "bf16[256, 64, 1, 1]" = torch.ops.aten.expand.default(empty_726, [256, 64, 1, 1]);  empty_726 = None
        view_3263: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_724, [512, 392, 512]);  empty_724 = None
        view_3264: "bf16[512, 64, 56, 56]" = torch.ops.aten.reshape.default(view_3263, [512, 64, 56, 56]);  view_3263 = None
        convolution_backward_99 = torch.ops.aten.convolution_backward.default(view_3262, view_3264, expand_100, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_3262 = view_3264 = expand_100 = None
        getitem_1277: "bf16[256, 64, 1, 1]" = convolution_backward_99[1];  convolution_backward_99 = None
        convert_element_type_312: "f32[256, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1277, torch.float32);  getitem_1277 = None
        
        # No stacktrace found for following nodes
        as_strided_default_10: "i8[102760448]" = torch.ops.aten.as_strided.default(full_default_339, [102760448], [1], 0)
        clone_default_5: "i8[102760448]" = torch.ops.aten.clone.default(as_strided_default_10);  as_strided_default_10 = None
        as_strided_default_11: "i8[200704, 512]" = torch.ops.aten.as_strided.default(clone_default_5, [200704, 512], [512, 1], 0);  clone_default_5 = None
        triton_kernel_wrapper_mutation_14 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 869, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_40, 'Y_ptr': as_strided_default_11, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_40 = triton_kernel_wrapper_mutation_14 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_3267: "i8[512, 392, 512]" = torch.ops.aten.reshape.default(as_strided_default_11, [512, 392, 512]);  as_strided_default_11 = None
        view_3268: "i8[512, 64, 56, 56]" = torch.ops.aten.reshape.default(view_3267, [512, 64, 56, 56]);  view_3267 = None
        mul_417: "bf16[512, 64, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_1273, view_3268);  getitem_1273 = view_3268 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_727: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_13 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 636, constant_args_idx = 870, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_35, 'S_ptr': getitem_36, 'M_ptr': getitem_37, 'Y_ptr': empty_727, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_35 = getitem_36 = getitem_37 = triton_kernel_wrapper_mutation_13 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_3283: "bf16[512, 64, 3136]" = torch.ops.aten.reshape.default(mul_417, [512, 64, 3136]);  mul_417 = None
        view_3284: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_727, [512, 392, 512]);  empty_727 = None
        view_3285: "bf16[512, 64, 3136]" = torch.ops.aten.reshape.default(view_3284, [512, 64, 3136]);  view_3284 = None
        
        # No stacktrace found for following nodes
        as_strided_default_6: "f32[64]" = torch.ops.aten.as_strided.default(full_default, [64], [1], 0)
        clone_default_3: "f32[64]" = torch.ops.aten.clone.default(as_strided_default_6);  as_strided_default_6 = None
        as_strided_default_7: "f32[64]" = torch.ops.aten.as_strided.default(clone_default_3, [64], [1], 0);  clone_default_3 = None
        as_strided_default_8: "f32[64]" = torch.ops.aten.as_strided.default(full_default, [64], [1], 0)
        clone_default_4: "f32[64]" = torch.ops.aten.clone.default(as_strided_default_8);  as_strided_default_8 = None
        as_strided_default_9: "f32[64]" = torch.ops.aten.as_strided.default(clone_default_4, [64], [1], 0);  clone_default_4 = None
        triton_kernel_wrapper_mutation_12 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 637, constant_args_idx = 871, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3285, 'DY': view_3283, 'DBETA': as_strided_default_7, 'DGAMMA': as_strided_default_9, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_12 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_728: "bf16[512, 64, 3136]" = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_207: "bf16[512, 64, 3136]" = torch.ops.aten.permute.default(empty_728, [0, 1, 2]);  empty_728 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_11 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 638, constant_args_idx = 872, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3285, 'DY': view_3283, 'INVSTD': rsqrt_2, 'GAMMA': primals_16, 'DBETA': as_strided_default_7, 'DGAMMA': as_strided_default_9, 'DX': permute_207, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024});  view_3285 = view_3283 = rsqrt_2 = primals_16 = triton_kernel_wrapper_mutation_11 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_729: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_10 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 639, constant_args_idx = 873, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_28, 'S_ptr': getitem_29, 'M_ptr': getitem_30, 'Y_ptr': empty_729, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_28 = getitem_29 = getitem_30 = triton_kernel_wrapper_mutation_10 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_3301: "bf16[512, 64, 56, 56]" = torch.ops.aten.reshape.default(permute_207, [512, 64, 56, 56]);  permute_207 = None
        empty_730: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_101: "bf16[512, 64, 56, 56]" = torch.ops.aten.expand.default(empty_730, [512, 64, 56, 56]);  empty_730 = None
        convolution_backward_100 = torch.ops.aten.convolution_backward.default(view_3301, expand_101, convert_element_type_3, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_101 = convert_element_type_3 = None
        getitem_1285: "bf16[512, 64, 56, 56]" = convolution_backward_100[0];  convolution_backward_100 = None
        empty_731: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_102: "bf16[64, 64, 3, 3]" = torch.ops.aten.expand.default(empty_731, [64, 64, 3, 3]);  empty_731 = None
        view_3302: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_729, [512, 392, 512]);  empty_729 = None
        view_3303: "bf16[512, 64, 56, 56]" = torch.ops.aten.reshape.default(view_3302, [512, 64, 56, 56]);  view_3302 = None
        convolution_backward_101 = torch.ops.aten.convolution_backward.default(view_3301, view_3303, expand_102, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_3301 = view_3303 = expand_102 = None
        getitem_1289: "bf16[64, 64, 3, 3]" = convolution_backward_101[1];  convolution_backward_101 = None
        convert_element_type_317: "f32[64, 64, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_1289, torch.float32);  getitem_1289 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_9 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 874, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_27, 'Y_ptr': full_default_339, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_27 = triton_kernel_wrapper_mutation_9 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_3306: "i8[512, 392, 512]" = torch.ops.aten.reshape.default(full_default_339, [512, 392, 512]);  full_default_339 = None
        view_3307: "i8[512, 64, 56, 56]" = torch.ops.aten.reshape.default(view_3306, [512, 64, 56, 56]);  view_3306 = None
        mul_418: "bf16[512, 64, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_1285, view_3307);  getitem_1285 = view_3307 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_732: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_8 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 640, constant_args_idx = 875, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_22, 'S_ptr': getitem_23, 'M_ptr': getitem_24, 'Y_ptr': empty_732, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_22 = getitem_23 = getitem_24 = triton_kernel_wrapper_mutation_8 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_3322: "bf16[512, 64, 3136]" = torch.ops.aten.reshape.default(mul_418, [512, 64, 3136]);  mul_418 = None
        view_3323: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_732, [512, 392, 512]);  empty_732 = None
        view_3324: "bf16[512, 64, 3136]" = torch.ops.aten.reshape.default(view_3323, [512, 64, 3136]);  view_3323 = None
        
        # No stacktrace found for following nodes
        as_strided_default_2: "f32[64]" = torch.ops.aten.as_strided.default(full_default, [64], [1], 0)
        clone_default_1: "f32[64]" = torch.ops.aten.clone.default(as_strided_default_2);  as_strided_default_2 = None
        as_strided_default_3: "f32[64]" = torch.ops.aten.as_strided.default(clone_default_1, [64], [1], 0);  clone_default_1 = None
        as_strided_default_4: "f32[64]" = torch.ops.aten.as_strided.default(full_default, [64], [1], 0)
        clone_default_2: "f32[64]" = torch.ops.aten.clone.default(as_strided_default_4);  as_strided_default_4 = None
        as_strided_default_5: "f32[64]" = torch.ops.aten.as_strided.default(clone_default_2, [64], [1], 0);  clone_default_2 = None
        triton_kernel_wrapper_mutation_7 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 641, constant_args_idx = 876, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3324, 'DY': view_3322, 'DBETA': as_strided_default_3, 'DGAMMA': as_strided_default_5, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_7 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_733: "bf16[512, 64, 3136]" = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_208: "bf16[512, 64, 3136]" = torch.ops.aten.permute.default(empty_733, [0, 1, 2]);  empty_733 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_6 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 642, constant_args_idx = 877, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3324, 'DY': view_3322, 'INVSTD': rsqrt_1, 'GAMMA': primals_10, 'DBETA': as_strided_default_3, 'DGAMMA': as_strided_default_5, 'DX': permute_208, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024});  view_3324 = view_3322 = rsqrt_1 = primals_10 = triton_kernel_wrapper_mutation_6 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_734: "bf16[200704, 512]" = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_5 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 643, constant_args_idx = 878, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_15, 'S_ptr': getitem_16, 'M_ptr': getitem_17, 'Y_ptr': empty_734, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_15 = getitem_16 = getitem_17 = triton_kernel_wrapper_mutation_5 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_3340: "bf16[512, 64, 56, 56]" = torch.ops.aten.reshape.default(permute_208, [512, 64, 56, 56]);  permute_208 = None
        empty_735: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_103: "bf16[512, 64, 56, 56]" = torch.ops.aten.expand.default(empty_735, [512, 64, 56, 56]);  empty_735 = None
        convolution_backward_102 = torch.ops.aten.convolution_backward.default(view_3340, expand_103, convert_element_type_2, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_103 = convert_element_type_2 = None
        getitem_1297: "bf16[512, 64, 56, 56]" = convolution_backward_102[0];  convolution_backward_102 = None
        empty_736: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_104: "bf16[64, 64, 1, 1]" = torch.ops.aten.expand.default(empty_736, [64, 64, 1, 1]);  empty_736 = None
        view_3341: "bf16[512, 392, 512]" = torch.ops.aten.reshape.default(empty_734, [512, 392, 512]);  empty_734 = None
        view_3342: "bf16[512, 64, 56, 56]" = torch.ops.aten.reshape.default(view_3341, [512, 64, 56, 56]);  view_3341 = None
        convolution_backward_103 = torch.ops.aten.convolution_backward.default(view_3340, view_3342, expand_104, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_3340 = view_3342 = expand_104 = None
        getitem_1301: "bf16[64, 64, 1, 1]" = convolution_backward_103[1];  convolution_backward_103 = None
        convert_element_type_322: "f32[64, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1301, torch.float32);  getitem_1301 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_243: "bf16[512, 64, 56, 56]" = torch.ops.aten.add.Tensor(getitem_1262, getitem_1297);  getitem_1262 = getitem_1297 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:8 in forward, code: maxpool = self.maxpool(relu);  relu = None
        _low_memory_max_pool_offsets_to_indices: "i64[512, 64, 56, 56]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_14, [3, 3], [112, 112], [2, 2], [1, 1], [1, 1]);  getitem_14 = None
        max_pool2d_with_indices_backward: "bf16[512, 64, 112, 112]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_243, getitem_10, [3, 3], [2, 2], [1, 1], [1, 1], False, _low_memory_max_pool_offsets_to_indices);  add_243 = getitem_10 = _low_memory_max_pool_offsets_to_indices = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_4 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 433, constant_args_idx = 879, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_12, 'Y_ptr': full_default_433, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16});  getitem_12 = triton_kernel_wrapper_mutation_4 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:174 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        view_3345: "i8[512, 1568, 512]" = torch.ops.aten.reshape.default(full_default_433, [512, 1568, 512]);  full_default_433 = None
        view_3346: "i8[512, 64, 112, 112]" = torch.ops.aten.reshape.default(view_3345, [512, 64, 112, 112]);  view_3345 = None
        mul_419: "bf16[512, 64, 112, 112]" = torch.ops.aten.mul.Tensor(max_pool2d_with_indices_backward, view_3346);  max_pool2d_with_indices_backward = view_3346 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_737: "bf16[802816, 512]" = torch.ops.aten.empty.memory_format([802816, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_3 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 644, constant_args_idx = 880, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_7, 'S_ptr': getitem_8, 'M_ptr': getitem_9, 'Y_ptr': empty_737, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem_7 = getitem_8 = getitem_9 = triton_kernel_wrapper_mutation_3 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        view_3361: "bf16[512, 64, 12544]" = torch.ops.aten.reshape.default(mul_419, [512, 64, 12544]);  mul_419 = None
        view_3362: "bf16[512, 1568, 512]" = torch.ops.aten.reshape.default(empty_737, [512, 1568, 512]);  empty_737 = None
        view_3363: "bf16[512, 64, 12544]" = torch.ops.aten.reshape.default(view_3362, [512, 64, 12544]);  view_3362 = None
        
        # No stacktrace found for following nodes
        as_strided_default: "f32[64]" = torch.ops.aten.as_strided.default(full_default, [64], [1], 0)
        clone_default: "f32[64]" = torch.ops.aten.clone.default(as_strided_default);  as_strided_default = None
        as_strided_default_1: "f32[64]" = torch.ops.aten.as_strided.default(clone_default, [64], [1], 0);  clone_default = None
        triton_kernel_wrapper_mutation_2 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 645, constant_args_idx = 881, grid = [(64, 6272, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3363, 'DY': view_3361, 'DBETA': full_default, 'DGAMMA': as_strided_default_1, 'M': 6422528, 'HW': 12544, 'stride_n': 802816, 'stride_c': 12544, 'BLOCK_M': 1024});  triton_kernel_wrapper_mutation_2 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:154 in forward, code: return _QuanBatchNorm.apply(
        empty_738: "bf16[512, 64, 12544]" = torch.ops.aten.empty.memory_format([512, 64, 12544], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_209: "bf16[512, 64, 12544]" = torch.ops.aten.permute.default(empty_738, [0, 1, 2]);  empty_738 = None
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation_1 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 646, constant_args_idx = 882, grid = [(64, 6272, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3363, 'DY': view_3361, 'INVSTD': rsqrt, 'GAMMA': primals_4, 'DBETA': full_default, 'DGAMMA': as_strided_default_1, 'DX': permute_209, 'M': 6422528, 'HW': 12544, 'stride_n': 802816, 'stride_c': 12544, 'BLOCK_M': 1024});  view_3363 = view_3361 = rsqrt = primals_4 = triton_kernel_wrapper_mutation_1 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_739: "bf16[150528, 512]" = torch.ops.aten.empty.memory_format([150528, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        triton_kernel_wrapper_mutation = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 647, constant_args_idx = 883, grid = [(150528, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem, 'S_ptr': getitem_1, 'M_ptr': getitem_2, 'Y_ptr': empty_739, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32});  getitem = getitem_1 = getitem_2 = triton_kernel_wrapper_mutation = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:74 in forward, code: return _QuanConv2d.apply(x, self.weight,
        view_3379: "bf16[512, 64, 112, 112]" = torch.ops.aten.reshape.default(permute_209, [512, 64, 112, 112]);  permute_209 = None
        empty_740: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_105: "bf16[64, 3, 7, 7]" = torch.ops.aten.expand.default(empty_740, [64, 3, 7, 7]);  empty_740 = None
        view_3380: "bf16[512, 294, 512]" = torch.ops.aten.reshape.default(empty_739, [512, 294, 512]);  empty_739 = None
        view_3381: "bf16[512, 3, 224, 224]" = torch.ops.aten.reshape.default(view_3380, [512, 3, 224, 224]);  view_3380 = None
        convolution_backward_104 = torch.ops.aten.convolution_backward.default(view_3379, view_3381, expand_105, None, [2, 2], [3, 3], [1, 1], False, [0], 1, [False, True, False]);  view_3379 = view_3381 = expand_105 = None
        getitem_1310: "bf16[64, 3, 7, 7]" = convolution_backward_104[1];  convolution_backward_104 = None
        convert_element_type_327: "f32[64, 3, 7, 7]" = torch.ops.prims.convert_element_type.default(getitem_1310, torch.float32);  getitem_1310 = None
        return (convert_element_type_327, None, None, as_strided_default_1, full_default, None, None, convert_element_type_322, None, as_strided_default_5, as_strided_default_3, None, None, convert_element_type_317, None, as_strided_default_9, as_strided_default_7, None, None, convert_element_type_312, None, as_strided_default_13, full_default_18, None, None, convert_element_type_307, None, as_strided_default_17, as_strided_default_15, None, None, convert_element_type_302, None, as_strided_default_23, as_strided_default_21, None, None, convert_element_type_297, None, as_strided_default_29, as_strided_default_27, None, None, convert_element_type_292, None, as_strided_default_35, as_strided_default_33, None, None, convert_element_type_287, None, as_strided_default_41, as_strided_default_39, None, None, convert_element_type_282, None, as_strided_default_47, as_strided_default_45, None, None, convert_element_type_277, None, as_strided_default_53, as_strided_default_51, None, None, convert_element_type_272, None, as_strided_default_57, full_default_64, None, None, convert_element_type_267, None, as_strided_default_61, as_strided_default_59, None, None, convert_element_type_262, None, as_strided_default_63, full_default_76, None, None, convert_element_type_257, None, as_strided_default_67, as_strided_default_65, None, None, convert_element_type_252, None, as_strided_default_73, as_strided_default_71, None, None, convert_element_type_247, None, as_strided_default_79, as_strided_default_77, None, None, convert_element_type_242, None, as_strided_default_85, as_strided_default_83, None, None, convert_element_type_237, None, as_strided_default_91, as_strided_default_89, None, None, convert_element_type_232, None, as_strided_default_97, as_strided_default_95, None, None, convert_element_type_227, None, as_strided_default_103, as_strided_default_101, None, None, convert_element_type_222, None, as_strided_default_109, as_strided_default_107, None, None, convert_element_type_217, None, as_strided_default_115, as_strided_default_113, None, None, convert_element_type_212, None, as_strided_default_121, as_strided_default_119, None, None, convert_element_type_207, None, as_strided_default_127, as_strided_default_125, None, None, convert_element_type_202, None, as_strided_default_133, as_strided_default_131, None, None, convert_element_type_197, None, as_strided_default_135, full_default_152, None, None, convert_element_type_192, None, as_strided_default_139, as_strided_default_137, None, None, convert_element_type_187, None, as_strided_default_145, as_strided_default_143, None, None, convert_element_type_182, None, as_strided_default_151, as_strided_default_149, None, None, convert_element_type_177, None, as_strided_default_157, as_strided_default_155, None, None, convert_element_type_172, None, as_strided_default_163, as_strided_default_161, None, None, convert_element_type_167, None, as_strided_default_169, as_strided_default_167, None, None, convert_element_type_162, None, as_strided_default_175, as_strided_default_173, None, None, convert_element_type_157, None, as_strided_default_181, as_strided_default_179, None, None, convert_element_type_152, None, as_strided_default_187, as_strided_default_185, None, None, convert_element_type_147, None, as_strided_default_193, as_strided_default_191, None, None, convert_element_type_142, None, as_strided_default_199, as_strided_default_197, None, None, convert_element_type_137, None, as_strided_default_205, as_strided_default_203, None, None, convert_element_type_132, None, as_strided_default_211, as_strided_default_209, None, None, convert_element_type_127, None, as_strided_default_217, as_strided_default_215, None, None, convert_element_type_122, None, as_strided_default_223, as_strided_default_221, None, None, convert_element_type_117, None, as_strided_default_229, as_strided_default_227, None, None, convert_element_type_112, None, as_strided_default_235, as_strided_default_233, None, None, convert_element_type_107, None, as_strided_default_241, as_strided_default_239, None, None, convert_element_type_102, None, as_strided_default_243, full_default_264, None, None, convert_element_type_97, None, as_strided_default_247, as_strided_default_245, None, None, convert_element_type_92, None, as_strided_default_253, as_strided_default_251, None, None, convert_element_type_87, None, as_strided_default_259, as_strided_default_257, None, None, convert_element_type_82, None, as_strided_default_265, as_strided_default_263, None, None, convert_element_type_77, None, as_strided_default_271, as_strided_default_269, None, None, convert_element_type_72, None, as_strided_default_277, as_strided_default_275, None, None, convert_element_type_67, None, as_strided_default_283, as_strided_default_281, None, None, convert_element_type_62)
        