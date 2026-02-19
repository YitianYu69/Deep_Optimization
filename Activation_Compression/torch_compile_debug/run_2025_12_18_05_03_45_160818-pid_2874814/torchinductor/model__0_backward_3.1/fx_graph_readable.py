class GraphModule(torch.nn.Module):
    def forward(self, primals_4: "f32[64]", primals_10: "f32[64]", primals_16: "f32[64]", primals_22: "f32[256]", primals_28: "f32[256]", primals_34: "f32[64]", primals_40: "f32[64]", primals_46: "f32[256]", primals_52: "f32[64]", primals_58: "f32[64]", primals_64: "f32[256]", primals_70: "f32[128]", primals_76: "f32[128]", primals_82: "f32[512]", primals_88: "f32[512]", primals_94: "f32[128]", primals_100: "f32[128]", primals_106: "f32[512]", primals_112: "f32[128]", primals_118: "f32[128]", primals_124: "f32[512]", primals_130: "f32[128]", primals_136: "f32[128]", primals_142: "f32[512]", primals_148: "f32[256]", primals_154: "f32[256]", primals_160: "f32[1024]", primals_166: "f32[1024]", primals_172: "f32[256]", primals_178: "f32[256]", primals_184: "f32[1024]", primals_190: "f32[256]", primals_196: "f32[256]", primals_202: "f32[1024]", primals_208: "f32[256]", primals_214: "f32[256]", primals_220: "f32[1024]", primals_226: "f32[256]", primals_232: "f32[256]", primals_238: "f32[1024]", primals_244: "f32[256]", primals_250: "f32[256]", primals_256: "f32[1024]", primals_262: "f32[512]", primals_268: "f32[512]", primals_274: "f32[2048]", primals_280: "f32[2048]", primals_286: "f32[512]", primals_292: "f32[512]", primals_298: "f32[2048]", primals_304: "f32[512]", primals_310: "f32[512]", primals_316: "f32[2048]", getitem: "i32[301056, 16]", getitem_1: "bf16[301056]", getitem_2: "bf16[301056]", rsqrt: "f32[64]", getitem_7: "i32[1605632, 16]", getitem_8: "bf16[1605632]", getitem_9: "bf16[1605632]", getitem_10: "bf16[512, 64, 112, 112]", getitem_12: "i32[1605632, 8]", getitem_14: "i8[512, 64, 56, 56]", convert_element_type_2: "bf16[64, 64, 1, 1]", getitem_15: "i32[401408, 16]", getitem_16: "bf16[401408]", getitem_17: "bf16[401408]", rsqrt_1: "f32[64]", getitem_22: "i32[401408, 16]", getitem_23: "bf16[401408]", getitem_24: "bf16[401408]", getitem_27: "i32[401408, 8]", convert_element_type_3: "bf16[64, 64, 3, 3]", getitem_28: "i32[401408, 16]", getitem_29: "bf16[401408]", getitem_30: "bf16[401408]", rsqrt_2: "f32[64]", getitem_35: "i32[401408, 16]", getitem_36: "bf16[401408]", getitem_37: "bf16[401408]", getitem_40: "i32[401408, 8]", convert_element_type_4: "bf16[256, 64, 1, 1]", getitem_41: "i32[401408, 16]", getitem_42: "bf16[401408]", getitem_43: "bf16[401408]", rsqrt_3: "f32[256]", getitem_48: "i32[1605632, 16]", getitem_49: "bf16[1605632]", getitem_50: "bf16[1605632]", convert_element_type_5: "bf16[256, 64, 1, 1]", getitem_51: "i32[401408, 16]", getitem_52: "bf16[401408]", getitem_53: "bf16[401408]", rsqrt_4: "f32[256]", getitem_58: "i32[1605632, 16]", getitem_59: "bf16[1605632]", getitem_60: "bf16[1605632]", getitem_63: "i32[1605632, 8]", convert_element_type_6: "bf16[64, 256, 1, 1]", getitem_64: "i32[1605632, 16]", getitem_65: "bf16[1605632]", getitem_66: "bf16[1605632]", rsqrt_5: "f32[64]", getitem_71: "i32[401408, 16]", getitem_72: "bf16[401408]", getitem_73: "bf16[401408]", getitem_76: "i32[401408, 8]", convert_element_type_7: "bf16[64, 64, 3, 3]", getitem_77: "i32[401408, 16]", getitem_78: "bf16[401408]", getitem_79: "bf16[401408]", rsqrt_6: "f32[64]", getitem_84: "i32[401408, 16]", getitem_85: "bf16[401408]", getitem_86: "bf16[401408]", getitem_89: "i32[401408, 8]", convert_element_type_8: "bf16[256, 64, 1, 1]", getitem_90: "i32[401408, 16]", getitem_91: "bf16[401408]", getitem_92: "bf16[401408]", rsqrt_7: "f32[256]", getitem_97: "i32[1605632, 16]", getitem_98: "bf16[1605632]", getitem_99: "bf16[1605632]", getitem_102: "i32[1605632, 8]", convert_element_type_9: "bf16[64, 256, 1, 1]", getitem_103: "i32[1605632, 16]", getitem_104: "bf16[1605632]", getitem_105: "bf16[1605632]", rsqrt_8: "f32[64]", getitem_110: "i32[401408, 16]", getitem_111: "bf16[401408]", getitem_112: "bf16[401408]", getitem_115: "i32[401408, 8]", convert_element_type_10: "bf16[64, 64, 3, 3]", getitem_116: "i32[401408, 16]", getitem_117: "bf16[401408]", getitem_118: "bf16[401408]", rsqrt_9: "f32[64]", getitem_123: "i32[401408, 16]", getitem_124: "bf16[401408]", getitem_125: "bf16[401408]", getitem_128: "i32[401408, 8]", convert_element_type_11: "bf16[256, 64, 1, 1]", getitem_129: "i32[401408, 16]", getitem_130: "bf16[401408]", getitem_131: "bf16[401408]", rsqrt_10: "f32[256]", getitem_136: "i32[1605632, 16]", getitem_137: "bf16[1605632]", getitem_138: "bf16[1605632]", getitem_141: "i32[1605632, 8]", convert_element_type_12: "bf16[128, 256, 1, 1]", getitem_142: "i32[1605632, 16]", getitem_143: "bf16[1605632]", getitem_144: "bf16[1605632]", rsqrt_11: "f32[128]", getitem_149: "i32[802816, 16]", getitem_150: "bf16[802816]", getitem_151: "bf16[802816]", getitem_154: "i32[802816, 8]", convert_element_type_13: "bf16[128, 128, 3, 3]", getitem_155: "i32[802816, 16]", getitem_156: "bf16[802816]", getitem_157: "bf16[802816]", rsqrt_12: "f32[128]", getitem_162: "i32[200704, 16]", getitem_163: "bf16[200704]", getitem_164: "bf16[200704]", getitem_167: "i32[200704, 8]", convert_element_type_14: "bf16[512, 128, 1, 1]", getitem_168: "i32[200704, 16]", getitem_169: "bf16[200704]", getitem_170: "bf16[200704]", rsqrt_13: "f32[512]", getitem_175: "i32[802816, 16]", getitem_176: "bf16[802816]", getitem_177: "bf16[802816]", convert_element_type_15: "bf16[512, 256, 1, 1]", getitem_178: "i32[1605632, 16]", getitem_179: "bf16[1605632]", getitem_180: "bf16[1605632]", rsqrt_14: "f32[512]", getitem_185: "i32[802816, 16]", getitem_186: "bf16[802816]", getitem_187: "bf16[802816]", getitem_190: "i32[802816, 8]", convert_element_type_16: "bf16[128, 512, 1, 1]", getitem_191: "i32[802816, 16]", getitem_192: "bf16[802816]", getitem_193: "bf16[802816]", rsqrt_15: "f32[128]", getitem_198: "i32[200704, 16]", getitem_199: "bf16[200704]", getitem_200: "bf16[200704]", getitem_203: "i32[200704, 8]", convert_element_type_17: "bf16[128, 128, 3, 3]", getitem_204: "i32[200704, 16]", getitem_205: "bf16[200704]", getitem_206: "bf16[200704]", rsqrt_16: "f32[128]", getitem_211: "i32[200704, 16]", getitem_212: "bf16[200704]", getitem_213: "bf16[200704]", getitem_216: "i32[200704, 8]", convert_element_type_18: "bf16[512, 128, 1, 1]", getitem_217: "i32[200704, 16]", getitem_218: "bf16[200704]", getitem_219: "bf16[200704]", rsqrt_17: "f32[512]", getitem_224: "i32[802816, 16]", getitem_225: "bf16[802816]", getitem_226: "bf16[802816]", getitem_229: "i32[802816, 8]", convert_element_type_19: "bf16[128, 512, 1, 1]", getitem_230: "i32[802816, 16]", getitem_231: "bf16[802816]", getitem_232: "bf16[802816]", rsqrt_18: "f32[128]", getitem_237: "i32[200704, 16]", getitem_238: "bf16[200704]", getitem_239: "bf16[200704]", getitem_242: "i32[200704, 8]", convert_element_type_20: "bf16[128, 128, 3, 3]", getitem_243: "i32[200704, 16]", getitem_244: "bf16[200704]", getitem_245: "bf16[200704]", rsqrt_19: "f32[128]", getitem_250: "i32[200704, 16]", getitem_251: "bf16[200704]", getitem_252: "bf16[200704]", getitem_255: "i32[200704, 8]", convert_element_type_21: "bf16[512, 128, 1, 1]", getitem_256: "i32[200704, 16]", getitem_257: "bf16[200704]", getitem_258: "bf16[200704]", rsqrt_20: "f32[512]", getitem_263: "i32[802816, 16]", getitem_264: "bf16[802816]", getitem_265: "bf16[802816]", getitem_268: "i32[802816, 8]", convert_element_type_22: "bf16[128, 512, 1, 1]", getitem_269: "i32[802816, 16]", getitem_270: "bf16[802816]", getitem_271: "bf16[802816]", rsqrt_21: "f32[128]", getitem_276: "i32[200704, 16]", getitem_277: "bf16[200704]", getitem_278: "bf16[200704]", getitem_281: "i32[200704, 8]", convert_element_type_23: "bf16[128, 128, 3, 3]", getitem_282: "i32[200704, 16]", getitem_283: "bf16[200704]", getitem_284: "bf16[200704]", rsqrt_22: "f32[128]", getitem_289: "i32[200704, 16]", getitem_290: "bf16[200704]", getitem_291: "bf16[200704]", getitem_294: "i32[200704, 8]", convert_element_type_24: "bf16[512, 128, 1, 1]", getitem_295: "i32[200704, 16]", getitem_296: "bf16[200704]", getitem_297: "bf16[200704]", rsqrt_23: "f32[512]", getitem_302: "i32[802816, 16]", getitem_303: "bf16[802816]", getitem_304: "bf16[802816]", getitem_307: "i32[802816, 8]", convert_element_type_25: "bf16[256, 512, 1, 1]", getitem_308: "i32[802816, 16]", getitem_309: "bf16[802816]", getitem_310: "bf16[802816]", rsqrt_24: "f32[256]", getitem_315: "i32[401408, 16]", getitem_316: "bf16[401408]", getitem_317: "bf16[401408]", getitem_320: "i32[401408, 8]", convert_element_type_26: "bf16[256, 256, 3, 3]", getitem_321: "i32[401408, 16]", getitem_322: "bf16[401408]", getitem_323: "bf16[401408]", rsqrt_25: "f32[256]", getitem_328: "i32[100352, 16]", getitem_329: "bf16[100352]", getitem_330: "bf16[100352]", getitem_333: "i32[100352, 8]", convert_element_type_27: "bf16[1024, 256, 1, 1]", getitem_334: "i32[100352, 16]", getitem_335: "bf16[100352]", getitem_336: "bf16[100352]", rsqrt_26: "f32[1024]", getitem_341: "i32[401408, 16]", getitem_342: "bf16[401408]", getitem_343: "bf16[401408]", convert_element_type_28: "bf16[1024, 512, 1, 1]", getitem_344: "i32[802816, 16]", getitem_345: "bf16[802816]", getitem_346: "bf16[802816]", rsqrt_27: "f32[1024]", getitem_351: "i32[401408, 16]", getitem_352: "bf16[401408]", getitem_353: "bf16[401408]", getitem_356: "i32[401408, 8]", convert_element_type_29: "bf16[256, 1024, 1, 1]", getitem_357: "i32[401408, 16]", getitem_358: "bf16[401408]", getitem_359: "bf16[401408]", rsqrt_28: "f32[256]", getitem_364: "i32[100352, 16]", getitem_365: "bf16[100352]", getitem_366: "bf16[100352]", getitem_369: "i32[100352, 8]", convert_element_type_30: "bf16[256, 256, 3, 3]", getitem_370: "i32[100352, 16]", getitem_371: "bf16[100352]", getitem_372: "bf16[100352]", rsqrt_29: "f32[256]", getitem_377: "i32[100352, 16]", getitem_378: "bf16[100352]", getitem_379: "bf16[100352]", getitem_382: "i32[100352, 8]", convert_element_type_31: "bf16[1024, 256, 1, 1]", getitem_383: "i32[100352, 16]", getitem_384: "bf16[100352]", getitem_385: "bf16[100352]", rsqrt_30: "f32[1024]", getitem_390: "i32[401408, 16]", getitem_391: "bf16[401408]", getitem_392: "bf16[401408]", getitem_395: "i32[401408, 8]", convert_element_type_32: "bf16[256, 1024, 1, 1]", getitem_396: "i32[401408, 16]", getitem_397: "bf16[401408]", getitem_398: "bf16[401408]", rsqrt_31: "f32[256]", getitem_403: "i32[100352, 16]", getitem_404: "bf16[100352]", getitem_405: "bf16[100352]", getitem_408: "i32[100352, 8]", convert_element_type_33: "bf16[256, 256, 3, 3]", getitem_409: "i32[100352, 16]", getitem_410: "bf16[100352]", getitem_411: "bf16[100352]", rsqrt_32: "f32[256]", getitem_416: "i32[100352, 16]", getitem_417: "bf16[100352]", getitem_418: "bf16[100352]", getitem_421: "i32[100352, 8]", convert_element_type_34: "bf16[1024, 256, 1, 1]", getitem_422: "i32[100352, 16]", getitem_423: "bf16[100352]", getitem_424: "bf16[100352]", rsqrt_33: "f32[1024]", getitem_429: "i32[401408, 16]", getitem_430: "bf16[401408]", getitem_431: "bf16[401408]", getitem_434: "i32[401408, 8]", convert_element_type_35: "bf16[256, 1024, 1, 1]", getitem_435: "i32[401408, 16]", getitem_436: "bf16[401408]", getitem_437: "bf16[401408]", rsqrt_34: "f32[256]", getitem_442: "i32[100352, 16]", getitem_443: "bf16[100352]", getitem_444: "bf16[100352]", getitem_447: "i32[100352, 8]", convert_element_type_36: "bf16[256, 256, 3, 3]", getitem_448: "i32[100352, 16]", getitem_449: "bf16[100352]", getitem_450: "bf16[100352]", rsqrt_35: "f32[256]", getitem_455: "i32[100352, 16]", getitem_456: "bf16[100352]", getitem_457: "bf16[100352]", getitem_460: "i32[100352, 8]", convert_element_type_37: "bf16[1024, 256, 1, 1]", getitem_461: "i32[100352, 16]", getitem_462: "bf16[100352]", getitem_463: "bf16[100352]", rsqrt_36: "f32[1024]", getitem_468: "i32[401408, 16]", getitem_469: "bf16[401408]", getitem_470: "bf16[401408]", getitem_473: "i32[401408, 8]", convert_element_type_38: "bf16[256, 1024, 1, 1]", getitem_474: "i32[401408, 16]", getitem_475: "bf16[401408]", getitem_476: "bf16[401408]", rsqrt_37: "f32[256]", getitem_481: "i32[100352, 16]", getitem_482: "bf16[100352]", getitem_483: "bf16[100352]", getitem_486: "i32[100352, 8]", convert_element_type_39: "bf16[256, 256, 3, 3]", getitem_487: "i32[100352, 16]", getitem_488: "bf16[100352]", getitem_489: "bf16[100352]", rsqrt_38: "f32[256]", getitem_494: "i32[100352, 16]", getitem_495: "bf16[100352]", getitem_496: "bf16[100352]", getitem_499: "i32[100352, 8]", convert_element_type_40: "bf16[1024, 256, 1, 1]", getitem_500: "i32[100352, 16]", getitem_501: "bf16[100352]", getitem_502: "bf16[100352]", rsqrt_39: "f32[1024]", getitem_507: "i32[401408, 16]", getitem_508: "bf16[401408]", getitem_509: "bf16[401408]", getitem_512: "i32[401408, 8]", convert_element_type_41: "bf16[256, 1024, 1, 1]", getitem_513: "i32[401408, 16]", getitem_514: "bf16[401408]", getitem_515: "bf16[401408]", rsqrt_40: "f32[256]", getitem_520: "i32[100352, 16]", getitem_521: "bf16[100352]", getitem_522: "bf16[100352]", getitem_525: "i32[100352, 8]", convert_element_type_42: "bf16[256, 256, 3, 3]", getitem_526: "i32[100352, 16]", getitem_527: "bf16[100352]", getitem_528: "bf16[100352]", rsqrt_41: "f32[256]", getitem_533: "i32[100352, 16]", getitem_534: "bf16[100352]", getitem_535: "bf16[100352]", getitem_538: "i32[100352, 8]", convert_element_type_43: "bf16[1024, 256, 1, 1]", getitem_539: "i32[100352, 16]", getitem_540: "bf16[100352]", getitem_541: "bf16[100352]", rsqrt_42: "f32[1024]", getitem_546: "i32[401408, 16]", getitem_547: "bf16[401408]", getitem_548: "bf16[401408]", getitem_551: "i32[401408, 8]", convert_element_type_44: "bf16[512, 1024, 1, 1]", getitem_552: "i32[401408, 16]", getitem_553: "bf16[401408]", getitem_554: "bf16[401408]", rsqrt_43: "f32[512]", getitem_559: "i32[200704, 16]", getitem_560: "bf16[200704]", getitem_561: "bf16[200704]", getitem_564: "i32[200704, 8]", convert_element_type_45: "bf16[512, 512, 3, 3]", getitem_565: "i32[200704, 16]", getitem_566: "bf16[200704]", getitem_567: "bf16[200704]", rsqrt_44: "f32[512]", getitem_572: "i32[50176, 16]", getitem_573: "bf16[50176]", getitem_574: "bf16[50176]", getitem_577: "i32[50176, 8]", convert_element_type_46: "bf16[2048, 512, 1, 1]", getitem_578: "i32[50176, 16]", getitem_579: "bf16[50176]", getitem_580: "bf16[50176]", rsqrt_45: "f32[2048]", getitem_585: "i32[200704, 16]", getitem_586: "bf16[200704]", getitem_587: "bf16[200704]", convert_element_type_47: "bf16[2048, 1024, 1, 1]", getitem_588: "i32[401408, 16]", getitem_589: "bf16[401408]", getitem_590: "bf16[401408]", rsqrt_46: "f32[2048]", getitem_595: "i32[200704, 16]", getitem_596: "bf16[200704]", getitem_597: "bf16[200704]", getitem_600: "i32[200704, 8]", convert_element_type_48: "bf16[512, 2048, 1, 1]", getitem_601: "i32[200704, 16]", getitem_602: "bf16[200704]", getitem_603: "bf16[200704]", rsqrt_47: "f32[512]", getitem_608: "i32[50176, 16]", getitem_609: "bf16[50176]", getitem_610: "bf16[50176]", getitem_613: "i32[50176, 8]", convert_element_type_49: "bf16[512, 512, 3, 3]", getitem_614: "i32[50176, 16]", getitem_615: "bf16[50176]", getitem_616: "bf16[50176]", rsqrt_48: "f32[512]", getitem_621: "i32[50176, 16]", getitem_622: "bf16[50176]", getitem_623: "bf16[50176]", getitem_626: "i32[50176, 8]", convert_element_type_50: "bf16[2048, 512, 1, 1]", getitem_627: "i32[50176, 16]", getitem_628: "bf16[50176]", getitem_629: "bf16[50176]", rsqrt_49: "f32[2048]", getitem_634: "i32[200704, 16]", getitem_635: "bf16[200704]", getitem_636: "bf16[200704]", getitem_639: "i32[200704, 8]", convert_element_type_51: "bf16[512, 2048, 1, 1]", getitem_640: "i32[200704, 16]", getitem_641: "bf16[200704]", getitem_642: "bf16[200704]", rsqrt_50: "f32[512]", getitem_647: "i32[50176, 16]", getitem_648: "bf16[50176]", getitem_649: "bf16[50176]", getitem_652: "i32[50176, 8]", convert_element_type_52: "bf16[512, 512, 3, 3]", getitem_653: "i32[50176, 16]", getitem_654: "bf16[50176]", getitem_655: "bf16[50176]", rsqrt_51: "f32[512]", getitem_660: "i32[50176, 16]", getitem_661: "bf16[50176]", getitem_662: "bf16[50176]", getitem_665: "i32[50176, 8]", convert_element_type_53: "bf16[2048, 512, 1, 1]", getitem_666: "i32[50176, 16]", getitem_667: "bf16[50176]", getitem_668: "bf16[50176]", rsqrt_52: "f32[2048]", getitem_673: "i32[200704, 16]", getitem_674: "bf16[200704]", getitem_675: "bf16[200704]", getitem_678: "i32[200704, 8]", getitem_679: "i32[4096, 16]", getitem_680: "bf16[4096]", getitem_681: "bf16[4096]", convert_element_type_54: "bf16[100, 2048]", tangents_1: "bf16[512, 100]"):
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:26 in forward, code: return _QuanLinear.apply(x, self.weight, self.quantizer, self.target_name, self.graph_mode, self.meta, self.qdrop)
        permute_156: "bf16[100, 512]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        empty_476: "bf16[4096, 256]" = torch.ops.aten.empty.memory_format([4096, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_311 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 432, constant_args_idx = 622, grid = [(4096, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_679, 'S_ptr': getitem_680, 'M_ptr': getitem_681, 'Y_ptr': empty_476, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_679 = getitem_680 = getitem_681 = empty_476 = None
        getitem_682: "bf16[4096, 256]" = triton_kernel_wrapper_functional_proxy_311['Y_ptr'];  triton_kernel_wrapper_functional_proxy_311 = None
        view_1639: "bf16[512, 8, 256]" = torch.ops.aten.view.default(getitem_682, [512, 8, 256]);  getitem_682 = None
        view_1640: "bf16[512, 2048]" = torch.ops.aten.view.default(view_1639, [512, -1]);  view_1639 = None
        mm_1: "bf16[100, 2048]" = torch.ops.aten.mm.default(permute_156, view_1640);  permute_156 = view_1640 = None
        mm_2: "bf16[512, 2048]" = torch.ops.aten.mm.default(tangents_1, convert_element_type_54);  tangents_1 = convert_element_type_54 = None
        convert_element_type_62: "f32[100, 2048]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        
         # File: <eval_with_key>.3 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:178 in forward, code: flatten = torch.flatten(avgpool, 1);  avgpool = None
        view_1644: "bf16[512, 2048, 1, 1]" = torch.ops.aten.view.default(mm_2, [512, 2048, 1, 1]);  mm_2 = None
        
         # File: <eval_with_key>.3 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:177 in forward, code: avgpool = self.avgpool(layer4_2_relu_2);  layer4_2_relu_2 = None
        expand: "bf16[512, 2048, 7, 7]" = torch.ops.aten.expand.default(view_1644, [512, 2048, 7, 7]);  view_1644 = None
        div_159: "bf16[512, 2048, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_310: "i8[200704, 256]" = torch.ops.aten.full.default([200704, 256], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_312 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 623, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_678, 'Y_ptr': full_default_310, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_678 = None
        getitem_683: "i8[200704, 256]" = triton_kernel_wrapper_functional_proxy_312['Y_ptr'];  triton_kernel_wrapper_functional_proxy_312 = None
        view_1648: "i8[512, 392, 256]" = torch.ops.aten.view.default(getitem_683, [512, 392, 256]);  getitem_683 = None
        view_1649: "i8[512, 100352]" = torch.ops.aten.view.default(view_1648, [512, -1]);  view_1648 = None
        view_1650: "i8[512, 2048, 7, 7]" = torch.ops.aten.view.default(view_1649, [512, 2048, 7, 7]);  view_1649 = None
        mul_371: "bf16[512, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(div_159, view_1650);  div_159 = view_1650 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_477: "bf16[200704, 256]" = torch.ops.aten.empty.memory_format([200704, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_313 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 434, constant_args_idx = 624, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_673, 'S_ptr': getitem_674, 'M_ptr': getitem_675, 'Y_ptr': empty_477, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_673 = getitem_674 = getitem_675 = empty_477 = None
        getitem_684: "bf16[200704, 256]" = triton_kernel_wrapper_functional_proxy_313['Y_ptr'];  triton_kernel_wrapper_functional_proxy_313 = None
        view_1666: "bf16[512, 2048, 49]" = torch.ops.aten.view.default(mul_371, [512, 2048, 49])
        view_1667: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_684, [512, 392, 256]);  getitem_684 = None
        view_1668: "bf16[512, 100352]" = torch.ops.aten.view.default(view_1667, [512, -1]);  view_1667 = None
        view_1669: "bf16[512, 2048, 49]" = torch.ops.aten.view.default(view_1668, [512, 2048, 49]);  view_1668 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        full_default_264: "f32[2048]" = torch.ops.aten.full.default([2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        triton_kernel_wrapper_functional_proxy_314 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 436, constant_args_idx = 625, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1669, 'DY': view_1666, 'DBETA': full_default_264, 'DGAMMA': full_default_264, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_685: "f32[2048]" = triton_kernel_wrapper_functional_proxy_314['DBETA']
        getitem_686: "f32[2048]" = triton_kernel_wrapper_functional_proxy_314['DGAMMA'];  triton_kernel_wrapper_functional_proxy_314 = None
        empty_478: "bf16[512, 2048, 49]" = torch.ops.aten.empty.memory_format([512, 2048, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_157: "bf16[512, 2048, 49]" = torch.ops.aten.permute.default(empty_478, [0, 1, 2]);  empty_478 = None
        triton_kernel_wrapper_functional_proxy_315 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 438, constant_args_idx = 626, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1669, 'DY': view_1666, 'INVSTD': rsqrt_52, 'GAMMA': primals_316, 'DBETA': getitem_685, 'DGAMMA': getitem_686, 'DX': permute_157, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_1669 = view_1666 = rsqrt_52 = primals_316 = permute_157 = None
        getitem_687: "bf16[512, 2048, 49]" = triton_kernel_wrapper_functional_proxy_315['DX'];  triton_kernel_wrapper_functional_proxy_315 = None
        convert_element_type_default_105: "f32[2048]" = torch.ops.prims.convert_element_type.default(getitem_686, torch.float32);  getitem_686 = None
        convert_element_type_default_104: "f32[2048]" = torch.ops.prims.convert_element_type.default(getitem_685, torch.float32);  getitem_685 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_479: "bf16[50176, 256]" = torch.ops.aten.empty.memory_format([50176, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_316 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 439, constant_args_idx = 627, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_666, 'S_ptr': getitem_667, 'M_ptr': getitem_668, 'Y_ptr': empty_479, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_666 = getitem_667 = getitem_668 = empty_479 = None
        getitem_688: "bf16[50176, 256]" = triton_kernel_wrapper_functional_proxy_316['Y_ptr'];  triton_kernel_wrapper_functional_proxy_316 = None
        view_1686: "bf16[512, 2048, 7, 7]" = torch.ops.aten.view.default(getitem_687, [512, 2048, 7, 7]);  getitem_687 = None
        empty_480: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_1: "bf16[512, 512, 7, 7]" = torch.ops.aten.expand.default(empty_480, [512, 512, 7, 7]);  empty_480 = None
        convolution_backward = torch.ops.aten.convolution_backward.default(view_1686, expand_1, convert_element_type_53, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_1 = convert_element_type_53 = None
        getitem_689: "bf16[512, 512, 7, 7]" = convolution_backward[0];  convolution_backward = None
        empty_481: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_2: "bf16[2048, 512, 1, 1]" = torch.ops.aten.expand.default(empty_481, [2048, 512, 1, 1]);  empty_481 = None
        view_1687: "bf16[512, 98, 256]" = torch.ops.aten.view.default(getitem_688, [512, 98, 256]);  getitem_688 = None
        view_1688: "bf16[512, 25088]" = torch.ops.aten.view.default(view_1687, [512, -1]);  view_1687 = None
        view_1689: "bf16[512, 512, 7, 7]" = torch.ops.aten.view.default(view_1688, [512, 512, 7, 7]);  view_1688 = None
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(view_1686, view_1689, expand_2, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1686 = view_1689 = expand_2 = None
        getitem_693: "bf16[2048, 512, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
        convert_element_type_67: "f32[2048, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_693, torch.float32);  getitem_693 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_313: "i8[50176, 256]" = torch.ops.aten.full.default([50176, 256], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_317 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 628, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_665, 'Y_ptr': full_default_313, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_665 = None
        getitem_695: "i8[50176, 256]" = triton_kernel_wrapper_functional_proxy_317['Y_ptr'];  triton_kernel_wrapper_functional_proxy_317 = None
        view_1693: "i8[512, 98, 256]" = torch.ops.aten.view.default(getitem_695, [512, 98, 256]);  getitem_695 = None
        view_1694: "i8[512, 25088]" = torch.ops.aten.view.default(view_1693, [512, -1]);  view_1693 = None
        view_1695: "i8[512, 512, 7, 7]" = torch.ops.aten.view.default(view_1694, [512, 512, 7, 7]);  view_1694 = None
        mul_372: "bf16[512, 512, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_689, view_1695);  getitem_689 = view_1695 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_482: "bf16[50176, 256]" = torch.ops.aten.empty.memory_format([50176, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_318 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 440, constant_args_idx = 629, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_660, 'S_ptr': getitem_661, 'M_ptr': getitem_662, 'Y_ptr': empty_482, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_660 = getitem_661 = getitem_662 = empty_482 = None
        getitem_696: "bf16[50176, 256]" = triton_kernel_wrapper_functional_proxy_318['Y_ptr'];  triton_kernel_wrapper_functional_proxy_318 = None
        view_1711: "bf16[512, 512, 49]" = torch.ops.aten.view.default(mul_372, [512, 512, 49]);  mul_372 = None
        view_1712: "bf16[512, 98, 256]" = torch.ops.aten.view.default(getitem_696, [512, 98, 256]);  getitem_696 = None
        view_1713: "bf16[512, 25088]" = torch.ops.aten.view.default(view_1712, [512, -1]);  view_1712 = None
        view_1714: "bf16[512, 512, 49]" = torch.ops.aten.view.default(view_1713, [512, 512, 49]);  view_1713 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        full_default_76: "f32[512]" = torch.ops.aten.full.default([512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        triton_kernel_wrapper_functional_proxy_319 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 441, constant_args_idx = 630, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1714, 'DY': view_1711, 'DBETA': full_default_76, 'DGAMMA': full_default_76, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_697: "f32[512]" = triton_kernel_wrapper_functional_proxy_319['DBETA']
        getitem_698: "f32[512]" = triton_kernel_wrapper_functional_proxy_319['DGAMMA'];  triton_kernel_wrapper_functional_proxy_319 = None
        empty_483: "bf16[512, 512, 49]" = torch.ops.aten.empty.memory_format([512, 512, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_158: "bf16[512, 512, 49]" = torch.ops.aten.permute.default(empty_483, [0, 1, 2]);  empty_483 = None
        triton_kernel_wrapper_functional_proxy_320 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 442, constant_args_idx = 631, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1714, 'DY': view_1711, 'INVSTD': rsqrt_51, 'GAMMA': primals_310, 'DBETA': getitem_697, 'DGAMMA': getitem_698, 'DX': permute_158, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_1714 = view_1711 = rsqrt_51 = primals_310 = permute_158 = None
        getitem_699: "bf16[512, 512, 49]" = triton_kernel_wrapper_functional_proxy_320['DX'];  triton_kernel_wrapper_functional_proxy_320 = None
        convert_element_type_default_103: "f32[512]" = torch.ops.prims.convert_element_type.default(getitem_698, torch.float32);  getitem_698 = None
        convert_element_type_default_102: "f32[512]" = torch.ops.prims.convert_element_type.default(getitem_697, torch.float32);  getitem_697 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_484: "bf16[50176, 256]" = torch.ops.aten.empty.memory_format([50176, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_321 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 443, constant_args_idx = 632, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_653, 'S_ptr': getitem_654, 'M_ptr': getitem_655, 'Y_ptr': empty_484, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_653 = getitem_654 = getitem_655 = empty_484 = None
        getitem_700: "bf16[50176, 256]" = triton_kernel_wrapper_functional_proxy_321['Y_ptr'];  triton_kernel_wrapper_functional_proxy_321 = None
        view_1731: "bf16[512, 512, 7, 7]" = torch.ops.aten.view.default(getitem_699, [512, 512, 7, 7]);  getitem_699 = None
        empty_485: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_3: "bf16[512, 512, 7, 7]" = torch.ops.aten.expand.default(empty_485, [512, 512, 7, 7]);  empty_485 = None
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(view_1731, expand_3, convert_element_type_52, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_3 = convert_element_type_52 = None
        getitem_701: "bf16[512, 512, 7, 7]" = convolution_backward_2[0];  convolution_backward_2 = None
        empty_486: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_4: "bf16[512, 512, 3, 3]" = torch.ops.aten.expand.default(empty_486, [512, 512, 3, 3]);  empty_486 = None
        view_1732: "bf16[512, 98, 256]" = torch.ops.aten.view.default(getitem_700, [512, 98, 256]);  getitem_700 = None
        view_1733: "bf16[512, 25088]" = torch.ops.aten.view.default(view_1732, [512, -1]);  view_1732 = None
        view_1734: "bf16[512, 512, 7, 7]" = torch.ops.aten.view.default(view_1733, [512, 512, 7, 7]);  view_1733 = None
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(view_1731, view_1734, expand_4, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_1731 = view_1734 = expand_4 = None
        getitem_705: "bf16[512, 512, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
        convert_element_type_72: "f32[512, 512, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_705, torch.float32);  getitem_705 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_322 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 633, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_652, 'Y_ptr': full_default_313, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_652 = None
        getitem_707: "i8[50176, 256]" = triton_kernel_wrapper_functional_proxy_322['Y_ptr'];  triton_kernel_wrapper_functional_proxy_322 = None
        view_1738: "i8[512, 98, 256]" = torch.ops.aten.view.default(getitem_707, [512, 98, 256]);  getitem_707 = None
        view_1739: "i8[512, 25088]" = torch.ops.aten.view.default(view_1738, [512, -1]);  view_1738 = None
        view_1740: "i8[512, 512, 7, 7]" = torch.ops.aten.view.default(view_1739, [512, 512, 7, 7]);  view_1739 = None
        mul_373: "bf16[512, 512, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_701, view_1740);  getitem_701 = view_1740 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_487: "bf16[50176, 256]" = torch.ops.aten.empty.memory_format([50176, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_323 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 444, constant_args_idx = 634, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_647, 'S_ptr': getitem_648, 'M_ptr': getitem_649, 'Y_ptr': empty_487, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_647 = getitem_648 = getitem_649 = empty_487 = None
        getitem_708: "bf16[50176, 256]" = triton_kernel_wrapper_functional_proxy_323['Y_ptr'];  triton_kernel_wrapper_functional_proxy_323 = None
        view_1756: "bf16[512, 512, 49]" = torch.ops.aten.view.default(mul_373, [512, 512, 49]);  mul_373 = None
        view_1757: "bf16[512, 98, 256]" = torch.ops.aten.view.default(getitem_708, [512, 98, 256]);  getitem_708 = None
        view_1758: "bf16[512, 25088]" = torch.ops.aten.view.default(view_1757, [512, -1]);  view_1757 = None
        view_1759: "bf16[512, 512, 49]" = torch.ops.aten.view.default(view_1758, [512, 512, 49]);  view_1758 = None
        triton_kernel_wrapper_functional_proxy_324 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 445, constant_args_idx = 635, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1759, 'DY': view_1756, 'DBETA': full_default_76, 'DGAMMA': full_default_76, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_709: "f32[512]" = triton_kernel_wrapper_functional_proxy_324['DBETA']
        getitem_710: "f32[512]" = triton_kernel_wrapper_functional_proxy_324['DGAMMA'];  triton_kernel_wrapper_functional_proxy_324 = None
        empty_488: "bf16[512, 512, 49]" = torch.ops.aten.empty.memory_format([512, 512, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_159: "bf16[512, 512, 49]" = torch.ops.aten.permute.default(empty_488, [0, 1, 2]);  empty_488 = None
        triton_kernel_wrapper_functional_proxy_325 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 446, constant_args_idx = 636, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1759, 'DY': view_1756, 'INVSTD': rsqrt_50, 'GAMMA': primals_304, 'DBETA': getitem_709, 'DGAMMA': getitem_710, 'DX': permute_159, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_1759 = view_1756 = rsqrt_50 = primals_304 = permute_159 = None
        getitem_711: "bf16[512, 512, 49]" = triton_kernel_wrapper_functional_proxy_325['DX'];  triton_kernel_wrapper_functional_proxy_325 = None
        convert_element_type_default_101: "f32[512]" = torch.ops.prims.convert_element_type.default(getitem_710, torch.float32);  getitem_710 = None
        convert_element_type_default_100: "f32[512]" = torch.ops.prims.convert_element_type.default(getitem_709, torch.float32);  getitem_709 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_489: "bf16[200704, 256]" = torch.ops.aten.empty.memory_format([200704, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_326 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 447, constant_args_idx = 637, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_640, 'S_ptr': getitem_641, 'M_ptr': getitem_642, 'Y_ptr': empty_489, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_640 = getitem_641 = getitem_642 = empty_489 = None
        getitem_712: "bf16[200704, 256]" = triton_kernel_wrapper_functional_proxy_326['Y_ptr'];  triton_kernel_wrapper_functional_proxy_326 = None
        view_1776: "bf16[512, 512, 7, 7]" = torch.ops.aten.view.default(getitem_711, [512, 512, 7, 7]);  getitem_711 = None
        empty_490: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_5: "bf16[512, 2048, 7, 7]" = torch.ops.aten.expand.default(empty_490, [512, 2048, 7, 7]);  empty_490 = None
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(view_1776, expand_5, convert_element_type_51, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_5 = convert_element_type_51 = None
        getitem_713: "bf16[512, 2048, 7, 7]" = convolution_backward_4[0];  convolution_backward_4 = None
        empty_491: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_6: "bf16[512, 2048, 1, 1]" = torch.ops.aten.expand.default(empty_491, [512, 2048, 1, 1]);  empty_491 = None
        view_1777: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_712, [512, 392, 256]);  getitem_712 = None
        view_1778: "bf16[512, 100352]" = torch.ops.aten.view.default(view_1777, [512, -1]);  view_1777 = None
        view_1779: "bf16[512, 2048, 7, 7]" = torch.ops.aten.view.default(view_1778, [512, 2048, 7, 7]);  view_1778 = None
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(view_1776, view_1779, expand_6, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1776 = view_1779 = expand_6 = None
        getitem_717: "bf16[512, 2048, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
        convert_element_type_77: "f32[512, 2048, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_717, torch.float32);  getitem_717 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_228: "bf16[512, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_371, getitem_713);  mul_371 = getitem_713 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_327 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 638, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_639, 'Y_ptr': full_default_310, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_639 = None
        getitem_719: "i8[200704, 256]" = triton_kernel_wrapper_functional_proxy_327['Y_ptr'];  triton_kernel_wrapper_functional_proxy_327 = None
        view_1783: "i8[512, 392, 256]" = torch.ops.aten.view.default(getitem_719, [512, 392, 256]);  getitem_719 = None
        view_1784: "i8[512, 100352]" = torch.ops.aten.view.default(view_1783, [512, -1]);  view_1783 = None
        view_1785: "i8[512, 2048, 7, 7]" = torch.ops.aten.view.default(view_1784, [512, 2048, 7, 7]);  view_1784 = None
        mul_374: "bf16[512, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(add_228, view_1785);  add_228 = view_1785 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_492: "bf16[200704, 256]" = torch.ops.aten.empty.memory_format([200704, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_328 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 448, constant_args_idx = 639, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_634, 'S_ptr': getitem_635, 'M_ptr': getitem_636, 'Y_ptr': empty_492, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_634 = getitem_635 = getitem_636 = empty_492 = None
        getitem_720: "bf16[200704, 256]" = triton_kernel_wrapper_functional_proxy_328['Y_ptr'];  triton_kernel_wrapper_functional_proxy_328 = None
        view_1801: "bf16[512, 2048, 49]" = torch.ops.aten.view.default(mul_374, [512, 2048, 49])
        view_1802: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_720, [512, 392, 256]);  getitem_720 = None
        view_1803: "bf16[512, 100352]" = torch.ops.aten.view.default(view_1802, [512, -1]);  view_1802 = None
        view_1804: "bf16[512, 2048, 49]" = torch.ops.aten.view.default(view_1803, [512, 2048, 49]);  view_1803 = None
        triton_kernel_wrapper_functional_proxy_329 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 449, constant_args_idx = 640, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1804, 'DY': view_1801, 'DBETA': full_default_264, 'DGAMMA': full_default_264, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_721: "f32[2048]" = triton_kernel_wrapper_functional_proxy_329['DBETA']
        getitem_722: "f32[2048]" = triton_kernel_wrapper_functional_proxy_329['DGAMMA'];  triton_kernel_wrapper_functional_proxy_329 = None
        empty_493: "bf16[512, 2048, 49]" = torch.ops.aten.empty.memory_format([512, 2048, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_160: "bf16[512, 2048, 49]" = torch.ops.aten.permute.default(empty_493, [0, 1, 2]);  empty_493 = None
        triton_kernel_wrapper_functional_proxy_330 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 450, constant_args_idx = 641, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1804, 'DY': view_1801, 'INVSTD': rsqrt_49, 'GAMMA': primals_298, 'DBETA': getitem_721, 'DGAMMA': getitem_722, 'DX': permute_160, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_1804 = view_1801 = rsqrt_49 = primals_298 = permute_160 = None
        getitem_723: "bf16[512, 2048, 49]" = triton_kernel_wrapper_functional_proxy_330['DX'];  triton_kernel_wrapper_functional_proxy_330 = None
        convert_element_type_default_99: "f32[2048]" = torch.ops.prims.convert_element_type.default(getitem_722, torch.float32);  getitem_722 = None
        convert_element_type_default_98: "f32[2048]" = torch.ops.prims.convert_element_type.default(getitem_721, torch.float32);  getitem_721 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_494: "bf16[50176, 256]" = torch.ops.aten.empty.memory_format([50176, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_331 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 451, constant_args_idx = 642, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_627, 'S_ptr': getitem_628, 'M_ptr': getitem_629, 'Y_ptr': empty_494, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_627 = getitem_628 = getitem_629 = empty_494 = None
        getitem_724: "bf16[50176, 256]" = triton_kernel_wrapper_functional_proxy_331['Y_ptr'];  triton_kernel_wrapper_functional_proxy_331 = None
        view_1821: "bf16[512, 2048, 7, 7]" = torch.ops.aten.view.default(getitem_723, [512, 2048, 7, 7]);  getitem_723 = None
        empty_495: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_7: "bf16[512, 512, 7, 7]" = torch.ops.aten.expand.default(empty_495, [512, 512, 7, 7]);  empty_495 = None
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(view_1821, expand_7, convert_element_type_50, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_7 = convert_element_type_50 = None
        getitem_725: "bf16[512, 512, 7, 7]" = convolution_backward_6[0];  convolution_backward_6 = None
        empty_496: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_8: "bf16[2048, 512, 1, 1]" = torch.ops.aten.expand.default(empty_496, [2048, 512, 1, 1]);  empty_496 = None
        view_1822: "bf16[512, 98, 256]" = torch.ops.aten.view.default(getitem_724, [512, 98, 256]);  getitem_724 = None
        view_1823: "bf16[512, 25088]" = torch.ops.aten.view.default(view_1822, [512, -1]);  view_1822 = None
        view_1824: "bf16[512, 512, 7, 7]" = torch.ops.aten.view.default(view_1823, [512, 512, 7, 7]);  view_1823 = None
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(view_1821, view_1824, expand_8, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1821 = view_1824 = expand_8 = None
        getitem_729: "bf16[2048, 512, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
        convert_element_type_82: "f32[2048, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_729, torch.float32);  getitem_729 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_332 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 643, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_626, 'Y_ptr': full_default_313, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_626 = None
        getitem_731: "i8[50176, 256]" = triton_kernel_wrapper_functional_proxy_332['Y_ptr'];  triton_kernel_wrapper_functional_proxy_332 = None
        view_1828: "i8[512, 98, 256]" = torch.ops.aten.view.default(getitem_731, [512, 98, 256]);  getitem_731 = None
        view_1829: "i8[512, 25088]" = torch.ops.aten.view.default(view_1828, [512, -1]);  view_1828 = None
        view_1830: "i8[512, 512, 7, 7]" = torch.ops.aten.view.default(view_1829, [512, 512, 7, 7]);  view_1829 = None
        mul_375: "bf16[512, 512, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_725, view_1830);  getitem_725 = view_1830 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_497: "bf16[50176, 256]" = torch.ops.aten.empty.memory_format([50176, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_333 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 452, constant_args_idx = 644, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_621, 'S_ptr': getitem_622, 'M_ptr': getitem_623, 'Y_ptr': empty_497, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_621 = getitem_622 = getitem_623 = empty_497 = None
        getitem_732: "bf16[50176, 256]" = triton_kernel_wrapper_functional_proxy_333['Y_ptr'];  triton_kernel_wrapper_functional_proxy_333 = None
        view_1846: "bf16[512, 512, 49]" = torch.ops.aten.view.default(mul_375, [512, 512, 49]);  mul_375 = None
        view_1847: "bf16[512, 98, 256]" = torch.ops.aten.view.default(getitem_732, [512, 98, 256]);  getitem_732 = None
        view_1848: "bf16[512, 25088]" = torch.ops.aten.view.default(view_1847, [512, -1]);  view_1847 = None
        view_1849: "bf16[512, 512, 49]" = torch.ops.aten.view.default(view_1848, [512, 512, 49]);  view_1848 = None
        triton_kernel_wrapper_functional_proxy_334 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 453, constant_args_idx = 645, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1849, 'DY': view_1846, 'DBETA': full_default_76, 'DGAMMA': full_default_76, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_733: "f32[512]" = triton_kernel_wrapper_functional_proxy_334['DBETA']
        getitem_734: "f32[512]" = triton_kernel_wrapper_functional_proxy_334['DGAMMA'];  triton_kernel_wrapper_functional_proxy_334 = None
        empty_498: "bf16[512, 512, 49]" = torch.ops.aten.empty.memory_format([512, 512, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_161: "bf16[512, 512, 49]" = torch.ops.aten.permute.default(empty_498, [0, 1, 2]);  empty_498 = None
        triton_kernel_wrapper_functional_proxy_335 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 454, constant_args_idx = 646, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1849, 'DY': view_1846, 'INVSTD': rsqrt_48, 'GAMMA': primals_292, 'DBETA': getitem_733, 'DGAMMA': getitem_734, 'DX': permute_161, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_1849 = view_1846 = rsqrt_48 = primals_292 = permute_161 = None
        getitem_735: "bf16[512, 512, 49]" = triton_kernel_wrapper_functional_proxy_335['DX'];  triton_kernel_wrapper_functional_proxy_335 = None
        convert_element_type_default_97: "f32[512]" = torch.ops.prims.convert_element_type.default(getitem_734, torch.float32);  getitem_734 = None
        convert_element_type_default_96: "f32[512]" = torch.ops.prims.convert_element_type.default(getitem_733, torch.float32);  getitem_733 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_499: "bf16[50176, 256]" = torch.ops.aten.empty.memory_format([50176, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_336 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 455, constant_args_idx = 647, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_614, 'S_ptr': getitem_615, 'M_ptr': getitem_616, 'Y_ptr': empty_499, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_614 = getitem_615 = getitem_616 = empty_499 = None
        getitem_736: "bf16[50176, 256]" = triton_kernel_wrapper_functional_proxy_336['Y_ptr'];  triton_kernel_wrapper_functional_proxy_336 = None
        view_1866: "bf16[512, 512, 7, 7]" = torch.ops.aten.view.default(getitem_735, [512, 512, 7, 7]);  getitem_735 = None
        empty_500: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_9: "bf16[512, 512, 7, 7]" = torch.ops.aten.expand.default(empty_500, [512, 512, 7, 7]);  empty_500 = None
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(view_1866, expand_9, convert_element_type_49, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_9 = convert_element_type_49 = None
        getitem_737: "bf16[512, 512, 7, 7]" = convolution_backward_8[0];  convolution_backward_8 = None
        empty_501: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_10: "bf16[512, 512, 3, 3]" = torch.ops.aten.expand.default(empty_501, [512, 512, 3, 3]);  empty_501 = None
        view_1867: "bf16[512, 98, 256]" = torch.ops.aten.view.default(getitem_736, [512, 98, 256]);  getitem_736 = None
        view_1868: "bf16[512, 25088]" = torch.ops.aten.view.default(view_1867, [512, -1]);  view_1867 = None
        view_1869: "bf16[512, 512, 7, 7]" = torch.ops.aten.view.default(view_1868, [512, 512, 7, 7]);  view_1868 = None
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(view_1866, view_1869, expand_10, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_1866 = view_1869 = expand_10 = None
        getitem_741: "bf16[512, 512, 3, 3]" = convolution_backward_9[1];  convolution_backward_9 = None
        convert_element_type_87: "f32[512, 512, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_741, torch.float32);  getitem_741 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_337 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 648, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_613, 'Y_ptr': full_default_313, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_613 = None
        getitem_743: "i8[50176, 256]" = triton_kernel_wrapper_functional_proxy_337['Y_ptr'];  triton_kernel_wrapper_functional_proxy_337 = None
        view_1873: "i8[512, 98, 256]" = torch.ops.aten.view.default(getitem_743, [512, 98, 256]);  getitem_743 = None
        view_1874: "i8[512, 25088]" = torch.ops.aten.view.default(view_1873, [512, -1]);  view_1873 = None
        view_1875: "i8[512, 512, 7, 7]" = torch.ops.aten.view.default(view_1874, [512, 512, 7, 7]);  view_1874 = None
        mul_376: "bf16[512, 512, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_737, view_1875);  getitem_737 = view_1875 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_502: "bf16[50176, 256]" = torch.ops.aten.empty.memory_format([50176, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_338 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 456, constant_args_idx = 649, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_608, 'S_ptr': getitem_609, 'M_ptr': getitem_610, 'Y_ptr': empty_502, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_608 = getitem_609 = getitem_610 = empty_502 = None
        getitem_744: "bf16[50176, 256]" = triton_kernel_wrapper_functional_proxy_338['Y_ptr'];  triton_kernel_wrapper_functional_proxy_338 = None
        view_1891: "bf16[512, 512, 49]" = torch.ops.aten.view.default(mul_376, [512, 512, 49]);  mul_376 = None
        view_1892: "bf16[512, 98, 256]" = torch.ops.aten.view.default(getitem_744, [512, 98, 256]);  getitem_744 = None
        view_1893: "bf16[512, 25088]" = torch.ops.aten.view.default(view_1892, [512, -1]);  view_1892 = None
        view_1894: "bf16[512, 512, 49]" = torch.ops.aten.view.default(view_1893, [512, 512, 49]);  view_1893 = None
        triton_kernel_wrapper_functional_proxy_339 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 457, constant_args_idx = 650, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1894, 'DY': view_1891, 'DBETA': full_default_76, 'DGAMMA': full_default_76, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_745: "f32[512]" = triton_kernel_wrapper_functional_proxy_339['DBETA']
        getitem_746: "f32[512]" = triton_kernel_wrapper_functional_proxy_339['DGAMMA'];  triton_kernel_wrapper_functional_proxy_339 = None
        empty_503: "bf16[512, 512, 49]" = torch.ops.aten.empty.memory_format([512, 512, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_162: "bf16[512, 512, 49]" = torch.ops.aten.permute.default(empty_503, [0, 1, 2]);  empty_503 = None
        triton_kernel_wrapper_functional_proxy_340 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 458, constant_args_idx = 651, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1894, 'DY': view_1891, 'INVSTD': rsqrt_47, 'GAMMA': primals_286, 'DBETA': getitem_745, 'DGAMMA': getitem_746, 'DX': permute_162, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_1894 = view_1891 = rsqrt_47 = primals_286 = permute_162 = None
        getitem_747: "bf16[512, 512, 49]" = triton_kernel_wrapper_functional_proxy_340['DX'];  triton_kernel_wrapper_functional_proxy_340 = None
        convert_element_type_default_95: "f32[512]" = torch.ops.prims.convert_element_type.default(getitem_746, torch.float32);  getitem_746 = None
        convert_element_type_default_94: "f32[512]" = torch.ops.prims.convert_element_type.default(getitem_745, torch.float32);  getitem_745 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_504: "bf16[200704, 256]" = torch.ops.aten.empty.memory_format([200704, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_341 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 459, constant_args_idx = 652, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_601, 'S_ptr': getitem_602, 'M_ptr': getitem_603, 'Y_ptr': empty_504, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_601 = getitem_602 = getitem_603 = empty_504 = None
        getitem_748: "bf16[200704, 256]" = triton_kernel_wrapper_functional_proxy_341['Y_ptr'];  triton_kernel_wrapper_functional_proxy_341 = None
        view_1911: "bf16[512, 512, 7, 7]" = torch.ops.aten.view.default(getitem_747, [512, 512, 7, 7]);  getitem_747 = None
        empty_505: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_11: "bf16[512, 2048, 7, 7]" = torch.ops.aten.expand.default(empty_505, [512, 2048, 7, 7]);  empty_505 = None
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(view_1911, expand_11, convert_element_type_48, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_11 = convert_element_type_48 = None
        getitem_749: "bf16[512, 2048, 7, 7]" = convolution_backward_10[0];  convolution_backward_10 = None
        empty_506: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_12: "bf16[512, 2048, 1, 1]" = torch.ops.aten.expand.default(empty_506, [512, 2048, 1, 1]);  empty_506 = None
        view_1912: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_748, [512, 392, 256]);  getitem_748 = None
        view_1913: "bf16[512, 100352]" = torch.ops.aten.view.default(view_1912, [512, -1]);  view_1912 = None
        view_1914: "bf16[512, 2048, 7, 7]" = torch.ops.aten.view.default(view_1913, [512, 2048, 7, 7]);  view_1913 = None
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(view_1911, view_1914, expand_12, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1911 = view_1914 = expand_12 = None
        getitem_753: "bf16[512, 2048, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
        convert_element_type_92: "f32[512, 2048, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_753, torch.float32);  getitem_753 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_229: "bf16[512, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_374, getitem_749);  mul_374 = getitem_749 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_342 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 653, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_600, 'Y_ptr': full_default_310, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_600 = None
        getitem_755: "i8[200704, 256]" = triton_kernel_wrapper_functional_proxy_342['Y_ptr'];  triton_kernel_wrapper_functional_proxy_342 = None
        view_1918: "i8[512, 392, 256]" = torch.ops.aten.view.default(getitem_755, [512, 392, 256]);  getitem_755 = None
        view_1919: "i8[512, 100352]" = torch.ops.aten.view.default(view_1918, [512, -1]);  view_1918 = None
        view_1920: "i8[512, 2048, 7, 7]" = torch.ops.aten.view.default(view_1919, [512, 2048, 7, 7]);  view_1919 = None
        mul_377: "bf16[512, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(add_229, view_1920);  add_229 = view_1920 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_507: "bf16[200704, 256]" = torch.ops.aten.empty.memory_format([200704, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_343 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 460, constant_args_idx = 654, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_595, 'S_ptr': getitem_596, 'M_ptr': getitem_597, 'Y_ptr': empty_507, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_595 = getitem_596 = getitem_597 = empty_507 = None
        getitem_756: "bf16[200704, 256]" = triton_kernel_wrapper_functional_proxy_343['Y_ptr'];  triton_kernel_wrapper_functional_proxy_343 = None
        view_1936: "bf16[512, 2048, 49]" = torch.ops.aten.view.default(mul_377, [512, 2048, 49]);  mul_377 = None
        view_1937: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_756, [512, 392, 256]);  getitem_756 = None
        view_1938: "bf16[512, 100352]" = torch.ops.aten.view.default(view_1937, [512, -1]);  view_1937 = None
        view_1939: "bf16[512, 2048, 49]" = torch.ops.aten.view.default(view_1938, [512, 2048, 49]);  view_1938 = None
        triton_kernel_wrapper_functional_proxy_344 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 461, constant_args_idx = 655, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1939, 'DY': view_1936, 'DBETA': full_default_264, 'DGAMMA': full_default_264, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_757: "f32[2048]" = triton_kernel_wrapper_functional_proxy_344['DBETA']
        getitem_758: "f32[2048]" = triton_kernel_wrapper_functional_proxy_344['DGAMMA'];  triton_kernel_wrapper_functional_proxy_344 = None
        empty_508: "bf16[512, 2048, 49]" = torch.ops.aten.empty.memory_format([512, 2048, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_163: "bf16[512, 2048, 49]" = torch.ops.aten.permute.default(empty_508, [0, 1, 2]);  empty_508 = None
        triton_kernel_wrapper_functional_proxy_345 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 462, constant_args_idx = 656, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1939, 'DY': view_1936, 'INVSTD': rsqrt_46, 'GAMMA': primals_280, 'DBETA': getitem_757, 'DGAMMA': getitem_758, 'DX': permute_163, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_1939 = rsqrt_46 = primals_280 = permute_163 = None
        getitem_759: "bf16[512, 2048, 49]" = triton_kernel_wrapper_functional_proxy_345['DX'];  triton_kernel_wrapper_functional_proxy_345 = None
        convert_element_type_default_93: "f32[2048]" = torch.ops.prims.convert_element_type.default(getitem_758, torch.float32);  getitem_758 = None
        convert_element_type_default_92: "f32[2048]" = torch.ops.prims.convert_element_type.default(getitem_757, torch.float32);  getitem_757 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_509: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_346 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 463, constant_args_idx = 657, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_588, 'S_ptr': getitem_589, 'M_ptr': getitem_590, 'Y_ptr': empty_509, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_588 = getitem_589 = getitem_590 = empty_509 = None
        getitem_760: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_346['Y_ptr'];  triton_kernel_wrapper_functional_proxy_346 = None
        view_1956: "bf16[512, 2048, 7, 7]" = torch.ops.aten.view.default(getitem_759, [512, 2048, 7, 7]);  getitem_759 = None
        empty_510: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_13: "bf16[512, 1024, 14, 14]" = torch.ops.aten.expand.default(empty_510, [512, 1024, 14, 14]);  empty_510 = None
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(view_1956, expand_13, convert_element_type_47, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_13 = convert_element_type_47 = None
        getitem_761: "bf16[512, 1024, 14, 14]" = convolution_backward_12[0];  convolution_backward_12 = None
        empty_511: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_14: "bf16[2048, 1024, 1, 1]" = torch.ops.aten.expand.default(empty_511, [2048, 1024, 1, 1]);  empty_511 = None
        view_1957: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_760, [512, 784, 256]);  getitem_760 = None
        view_1958: "bf16[512, 200704]" = torch.ops.aten.view.default(view_1957, [512, -1]);  view_1957 = None
        view_1959: "bf16[512, 1024, 14, 14]" = torch.ops.aten.view.default(view_1958, [512, 1024, 14, 14]);  view_1958 = None
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(view_1956, view_1959, expand_14, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1956 = view_1959 = expand_14 = None
        getitem_765: "bf16[2048, 1024, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
        convert_element_type_97: "f32[2048, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_765, torch.float32);  getitem_765 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_512: "bf16[200704, 256]" = torch.ops.aten.empty.memory_format([200704, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_347 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 464, constant_args_idx = 658, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_585, 'S_ptr': getitem_586, 'M_ptr': getitem_587, 'Y_ptr': empty_512, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_585 = getitem_586 = getitem_587 = empty_512 = None
        getitem_767: "bf16[200704, 256]" = triton_kernel_wrapper_functional_proxy_347['Y_ptr'];  triton_kernel_wrapper_functional_proxy_347 = None
        view_1976: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_767, [512, 392, 256]);  getitem_767 = None
        view_1977: "bf16[512, 100352]" = torch.ops.aten.view.default(view_1976, [512, -1]);  view_1976 = None
        view_1978: "bf16[512, 2048, 49]" = torch.ops.aten.view.default(view_1977, [512, 2048, 49]);  view_1977 = None
        triton_kernel_wrapper_functional_proxy_348 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 465, constant_args_idx = 659, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1978, 'DY': view_1936, 'DBETA': full_default_264, 'DGAMMA': full_default_264, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA']);  full_default_264 = None
        getitem_768: "f32[2048]" = triton_kernel_wrapper_functional_proxy_348['DBETA']
        getitem_769: "f32[2048]" = triton_kernel_wrapper_functional_proxy_348['DGAMMA'];  triton_kernel_wrapper_functional_proxy_348 = None
        empty_513: "bf16[512, 2048, 49]" = torch.ops.aten.empty.memory_format([512, 2048, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_164: "bf16[512, 2048, 49]" = torch.ops.aten.permute.default(empty_513, [0, 1, 2]);  empty_513 = None
        triton_kernel_wrapper_functional_proxy_349 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 466, constant_args_idx = 660, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1978, 'DY': view_1936, 'INVSTD': rsqrt_45, 'GAMMA': primals_274, 'DBETA': getitem_768, 'DGAMMA': getitem_769, 'DX': permute_164, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_1978 = view_1936 = rsqrt_45 = primals_274 = permute_164 = None
        getitem_770: "bf16[512, 2048, 49]" = triton_kernel_wrapper_functional_proxy_349['DX'];  triton_kernel_wrapper_functional_proxy_349 = None
        convert_element_type_default_91: "f32[2048]" = torch.ops.prims.convert_element_type.default(getitem_769, torch.float32);  getitem_769 = None
        convert_element_type_default_90: "f32[2048]" = torch.ops.prims.convert_element_type.default(getitem_768, torch.float32);  getitem_768 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_514: "bf16[50176, 256]" = torch.ops.aten.empty.memory_format([50176, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_350 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 467, constant_args_idx = 661, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_578, 'S_ptr': getitem_579, 'M_ptr': getitem_580, 'Y_ptr': empty_514, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_578 = getitem_579 = getitem_580 = empty_514 = None
        getitem_771: "bf16[50176, 256]" = triton_kernel_wrapper_functional_proxy_350['Y_ptr'];  triton_kernel_wrapper_functional_proxy_350 = None
        view_1995: "bf16[512, 2048, 7, 7]" = torch.ops.aten.view.default(getitem_770, [512, 2048, 7, 7]);  getitem_770 = None
        empty_515: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_15: "bf16[512, 512, 7, 7]" = torch.ops.aten.expand.default(empty_515, [512, 512, 7, 7]);  empty_515 = None
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(view_1995, expand_15, convert_element_type_46, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_15 = convert_element_type_46 = None
        getitem_772: "bf16[512, 512, 7, 7]" = convolution_backward_14[0];  convolution_backward_14 = None
        empty_516: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_16: "bf16[2048, 512, 1, 1]" = torch.ops.aten.expand.default(empty_516, [2048, 512, 1, 1]);  empty_516 = None
        view_1996: "bf16[512, 98, 256]" = torch.ops.aten.view.default(getitem_771, [512, 98, 256]);  getitem_771 = None
        view_1997: "bf16[512, 25088]" = torch.ops.aten.view.default(view_1996, [512, -1]);  view_1996 = None
        view_1998: "bf16[512, 512, 7, 7]" = torch.ops.aten.view.default(view_1997, [512, 512, 7, 7]);  view_1997 = None
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(view_1995, view_1998, expand_16, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1995 = view_1998 = expand_16 = None
        getitem_776: "bf16[2048, 512, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
        convert_element_type_102: "f32[2048, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_776, torch.float32);  getitem_776 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_351 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 662, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_577, 'Y_ptr': full_default_313, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_577 = full_default_313 = None
        getitem_778: "i8[50176, 256]" = triton_kernel_wrapper_functional_proxy_351['Y_ptr'];  triton_kernel_wrapper_functional_proxy_351 = None
        view_2002: "i8[512, 98, 256]" = torch.ops.aten.view.default(getitem_778, [512, 98, 256]);  getitem_778 = None
        view_2003: "i8[512, 25088]" = torch.ops.aten.view.default(view_2002, [512, -1]);  view_2002 = None
        view_2004: "i8[512, 512, 7, 7]" = torch.ops.aten.view.default(view_2003, [512, 512, 7, 7]);  view_2003 = None
        mul_378: "bf16[512, 512, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_772, view_2004);  getitem_772 = view_2004 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_517: "bf16[50176, 256]" = torch.ops.aten.empty.memory_format([50176, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_352 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 468, constant_args_idx = 663, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_572, 'S_ptr': getitem_573, 'M_ptr': getitem_574, 'Y_ptr': empty_517, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_572 = getitem_573 = getitem_574 = empty_517 = None
        getitem_779: "bf16[50176, 256]" = triton_kernel_wrapper_functional_proxy_352['Y_ptr'];  triton_kernel_wrapper_functional_proxy_352 = None
        view_2020: "bf16[512, 512, 49]" = torch.ops.aten.view.default(mul_378, [512, 512, 49]);  mul_378 = None
        view_2021: "bf16[512, 98, 256]" = torch.ops.aten.view.default(getitem_779, [512, 98, 256]);  getitem_779 = None
        view_2022: "bf16[512, 25088]" = torch.ops.aten.view.default(view_2021, [512, -1]);  view_2021 = None
        view_2023: "bf16[512, 512, 49]" = torch.ops.aten.view.default(view_2022, [512, 512, 49]);  view_2022 = None
        triton_kernel_wrapper_functional_proxy_353 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 469, constant_args_idx = 664, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2023, 'DY': view_2020, 'DBETA': full_default_76, 'DGAMMA': full_default_76, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_780: "f32[512]" = triton_kernel_wrapper_functional_proxy_353['DBETA']
        getitem_781: "f32[512]" = triton_kernel_wrapper_functional_proxy_353['DGAMMA'];  triton_kernel_wrapper_functional_proxy_353 = None
        empty_518: "bf16[512, 512, 49]" = torch.ops.aten.empty.memory_format([512, 512, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_165: "bf16[512, 512, 49]" = torch.ops.aten.permute.default(empty_518, [0, 1, 2]);  empty_518 = None
        triton_kernel_wrapper_functional_proxy_354 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 470, constant_args_idx = 665, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2023, 'DY': view_2020, 'INVSTD': rsqrt_44, 'GAMMA': primals_268, 'DBETA': getitem_780, 'DGAMMA': getitem_781, 'DX': permute_165, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2023 = view_2020 = rsqrt_44 = primals_268 = permute_165 = None
        getitem_782: "bf16[512, 512, 49]" = triton_kernel_wrapper_functional_proxy_354['DX'];  triton_kernel_wrapper_functional_proxy_354 = None
        convert_element_type_default_89: "f32[512]" = torch.ops.prims.convert_element_type.default(getitem_781, torch.float32);  getitem_781 = None
        convert_element_type_default_88: "f32[512]" = torch.ops.prims.convert_element_type.default(getitem_780, torch.float32);  getitem_780 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_519: "bf16[200704, 256]" = torch.ops.aten.empty.memory_format([200704, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_355 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 471, constant_args_idx = 666, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_565, 'S_ptr': getitem_566, 'M_ptr': getitem_567, 'Y_ptr': empty_519, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_565 = getitem_566 = getitem_567 = empty_519 = None
        getitem_783: "bf16[200704, 256]" = triton_kernel_wrapper_functional_proxy_355['Y_ptr'];  triton_kernel_wrapper_functional_proxy_355 = None
        view_2040: "bf16[512, 512, 7, 7]" = torch.ops.aten.view.default(getitem_782, [512, 512, 7, 7]);  getitem_782 = None
        empty_520: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_17: "bf16[512, 512, 14, 14]" = torch.ops.aten.expand.default(empty_520, [512, 512, 14, 14]);  empty_520 = None
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(view_2040, expand_17, convert_element_type_45, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_17 = convert_element_type_45 = None
        getitem_784: "bf16[512, 512, 14, 14]" = convolution_backward_16[0];  convolution_backward_16 = None
        empty_521: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_18: "bf16[512, 512, 3, 3]" = torch.ops.aten.expand.default(empty_521, [512, 512, 3, 3]);  empty_521 = None
        view_2041: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_783, [512, 392, 256]);  getitem_783 = None
        view_2042: "bf16[512, 100352]" = torch.ops.aten.view.default(view_2041, [512, -1]);  view_2041 = None
        view_2043: "bf16[512, 512, 14, 14]" = torch.ops.aten.view.default(view_2042, [512, 512, 14, 14]);  view_2042 = None
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(view_2040, view_2043, expand_18, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_2040 = view_2043 = expand_18 = None
        getitem_788: "bf16[512, 512, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None
        convert_element_type_107: "f32[512, 512, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_788, torch.float32);  getitem_788 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_356 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 667, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_564, 'Y_ptr': full_default_310, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_564 = None
        getitem_790: "i8[200704, 256]" = triton_kernel_wrapper_functional_proxy_356['Y_ptr'];  triton_kernel_wrapper_functional_proxy_356 = None
        view_2047: "i8[512, 392, 256]" = torch.ops.aten.view.default(getitem_790, [512, 392, 256]);  getitem_790 = None
        view_2048: "i8[512, 100352]" = torch.ops.aten.view.default(view_2047, [512, -1]);  view_2047 = None
        view_2049: "i8[512, 512, 14, 14]" = torch.ops.aten.view.default(view_2048, [512, 512, 14, 14]);  view_2048 = None
        mul_379: "bf16[512, 512, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_784, view_2049);  getitem_784 = view_2049 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_522: "bf16[200704, 256]" = torch.ops.aten.empty.memory_format([200704, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_357 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 472, constant_args_idx = 668, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_559, 'S_ptr': getitem_560, 'M_ptr': getitem_561, 'Y_ptr': empty_522, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_559 = getitem_560 = getitem_561 = empty_522 = None
        getitem_791: "bf16[200704, 256]" = triton_kernel_wrapper_functional_proxy_357['Y_ptr'];  triton_kernel_wrapper_functional_proxy_357 = None
        view_2065: "bf16[512, 512, 196]" = torch.ops.aten.view.default(mul_379, [512, 512, 196]);  mul_379 = None
        view_2066: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_791, [512, 392, 256]);  getitem_791 = None
        view_2067: "bf16[512, 100352]" = torch.ops.aten.view.default(view_2066, [512, -1]);  view_2066 = None
        view_2068: "bf16[512, 512, 196]" = torch.ops.aten.view.default(view_2067, [512, 512, 196]);  view_2067 = None
        triton_kernel_wrapper_functional_proxy_358 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 473, constant_args_idx = 669, grid = [(512, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2068, 'DY': view_2065, 'DBETA': full_default_76, 'DGAMMA': full_default_76, 'M': 100352, 'HW': 196, 'stride_n': 100352, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_792: "f32[512]" = triton_kernel_wrapper_functional_proxy_358['DBETA']
        getitem_793: "f32[512]" = triton_kernel_wrapper_functional_proxy_358['DGAMMA'];  triton_kernel_wrapper_functional_proxy_358 = None
        empty_523: "bf16[512, 512, 196]" = torch.ops.aten.empty.memory_format([512, 512, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_166: "bf16[512, 512, 196]" = torch.ops.aten.permute.default(empty_523, [0, 1, 2]);  empty_523 = None
        triton_kernel_wrapper_functional_proxy_359 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 474, constant_args_idx = 670, grid = [(512, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2068, 'DY': view_2065, 'INVSTD': rsqrt_43, 'GAMMA': primals_262, 'DBETA': getitem_792, 'DGAMMA': getitem_793, 'DX': permute_166, 'M': 100352, 'HW': 196, 'stride_n': 100352, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2068 = view_2065 = rsqrt_43 = primals_262 = permute_166 = None
        getitem_794: "bf16[512, 512, 196]" = triton_kernel_wrapper_functional_proxy_359['DX'];  triton_kernel_wrapper_functional_proxy_359 = None
        convert_element_type_default_87: "f32[512]" = torch.ops.prims.convert_element_type.default(getitem_793, torch.float32);  getitem_793 = None
        convert_element_type_default_86: "f32[512]" = torch.ops.prims.convert_element_type.default(getitem_792, torch.float32);  getitem_792 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_524: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_360 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 475, constant_args_idx = 671, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_552, 'S_ptr': getitem_553, 'M_ptr': getitem_554, 'Y_ptr': empty_524, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_552 = getitem_553 = getitem_554 = empty_524 = None
        getitem_795: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_360['Y_ptr'];  triton_kernel_wrapper_functional_proxy_360 = None
        view_2085: "bf16[512, 512, 14, 14]" = torch.ops.aten.view.default(getitem_794, [512, 512, 14, 14]);  getitem_794 = None
        empty_525: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_19: "bf16[512, 1024, 14, 14]" = torch.ops.aten.expand.default(empty_525, [512, 1024, 14, 14]);  empty_525 = None
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(view_2085, expand_19, convert_element_type_44, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_19 = convert_element_type_44 = None
        getitem_796: "bf16[512, 1024, 14, 14]" = convolution_backward_18[0];  convolution_backward_18 = None
        empty_526: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_20: "bf16[512, 1024, 1, 1]" = torch.ops.aten.expand.default(empty_526, [512, 1024, 1, 1]);  empty_526 = None
        view_2086: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_795, [512, 784, 256]);  getitem_795 = None
        view_2087: "bf16[512, 200704]" = torch.ops.aten.view.default(view_2086, [512, -1]);  view_2086 = None
        view_2088: "bf16[512, 1024, 14, 14]" = torch.ops.aten.view.default(view_2087, [512, 1024, 14, 14]);  view_2087 = None
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(view_2085, view_2088, expand_20, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2085 = view_2088 = expand_20 = None
        getitem_800: "bf16[512, 1024, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
        convert_element_type_112: "f32[512, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_800, torch.float32);  getitem_800 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_230: "bf16[512, 1024, 14, 14]" = torch.ops.aten.add.Tensor(getitem_761, getitem_796);  getitem_761 = getitem_796 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_339: "i8[401408, 256]" = torch.ops.aten.full.default([401408, 256], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_361 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 672, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_551, 'Y_ptr': full_default_339, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_551 = None
        getitem_802: "i8[401408, 256]" = triton_kernel_wrapper_functional_proxy_361['Y_ptr'];  triton_kernel_wrapper_functional_proxy_361 = None
        view_2092: "i8[512, 784, 256]" = torch.ops.aten.view.default(getitem_802, [512, 784, 256]);  getitem_802 = None
        view_2093: "i8[512, 200704]" = torch.ops.aten.view.default(view_2092, [512, -1]);  view_2092 = None
        view_2094: "i8[512, 1024, 14, 14]" = torch.ops.aten.view.default(view_2093, [512, 1024, 14, 14]);  view_2093 = None
        mul_380: "bf16[512, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(add_230, view_2094);  add_230 = view_2094 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_527: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_362 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 476, constant_args_idx = 673, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_546, 'S_ptr': getitem_547, 'M_ptr': getitem_548, 'Y_ptr': empty_527, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_546 = getitem_547 = getitem_548 = empty_527 = None
        getitem_803: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_362['Y_ptr'];  triton_kernel_wrapper_functional_proxy_362 = None
        view_2110: "bf16[512, 1024, 196]" = torch.ops.aten.view.default(mul_380, [512, 1024, 196])
        view_2111: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_803, [512, 784, 256]);  getitem_803 = None
        view_2112: "bf16[512, 200704]" = torch.ops.aten.view.default(view_2111, [512, -1]);  view_2111 = None
        view_2113: "bf16[512, 1024, 196]" = torch.ops.aten.view.default(view_2112, [512, 1024, 196]);  view_2112 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        full_default_152: "f32[1024]" = torch.ops.aten.full.default([1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        triton_kernel_wrapper_functional_proxy_363 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 477, constant_args_idx = 674, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2113, 'DY': view_2110, 'DBETA': full_default_152, 'DGAMMA': full_default_152, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_804: "f32[1024]" = triton_kernel_wrapper_functional_proxy_363['DBETA']
        getitem_805: "f32[1024]" = triton_kernel_wrapper_functional_proxy_363['DGAMMA'];  triton_kernel_wrapper_functional_proxy_363 = None
        empty_528: "bf16[512, 1024, 196]" = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_167: "bf16[512, 1024, 196]" = torch.ops.aten.permute.default(empty_528, [0, 1, 2]);  empty_528 = None
        triton_kernel_wrapper_functional_proxy_364 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 478, constant_args_idx = 675, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2113, 'DY': view_2110, 'INVSTD': rsqrt_42, 'GAMMA': primals_256, 'DBETA': getitem_804, 'DGAMMA': getitem_805, 'DX': permute_167, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2113 = view_2110 = rsqrt_42 = primals_256 = permute_167 = None
        getitem_806: "bf16[512, 1024, 196]" = triton_kernel_wrapper_functional_proxy_364['DX'];  triton_kernel_wrapper_functional_proxy_364 = None
        convert_element_type_default_85: "f32[1024]" = torch.ops.prims.convert_element_type.default(getitem_805, torch.float32);  getitem_805 = None
        convert_element_type_default_84: "f32[1024]" = torch.ops.prims.convert_element_type.default(getitem_804, torch.float32);  getitem_804 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_529: "bf16[100352, 256]" = torch.ops.aten.empty.memory_format([100352, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_365 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 479, constant_args_idx = 676, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_539, 'S_ptr': getitem_540, 'M_ptr': getitem_541, 'Y_ptr': empty_529, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_539 = getitem_540 = getitem_541 = empty_529 = None
        getitem_807: "bf16[100352, 256]" = triton_kernel_wrapper_functional_proxy_365['Y_ptr'];  triton_kernel_wrapper_functional_proxy_365 = None
        view_2130: "bf16[512, 1024, 14, 14]" = torch.ops.aten.view.default(getitem_806, [512, 1024, 14, 14]);  getitem_806 = None
        empty_530: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_21: "bf16[512, 256, 14, 14]" = torch.ops.aten.expand.default(empty_530, [512, 256, 14, 14]);  empty_530 = None
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(view_2130, expand_21, convert_element_type_43, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_21 = convert_element_type_43 = None
        getitem_808: "bf16[512, 256, 14, 14]" = convolution_backward_20[0];  convolution_backward_20 = None
        empty_531: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_22: "bf16[1024, 256, 1, 1]" = torch.ops.aten.expand.default(empty_531, [1024, 256, 1, 1]);  empty_531 = None
        view_2131: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_807, [512, 196, 256]);  getitem_807 = None
        view_2132: "bf16[512, 50176]" = torch.ops.aten.view.default(view_2131, [512, -1]);  view_2131 = None
        view_2133: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(view_2132, [512, 256, 14, 14]);  view_2132 = None
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(view_2130, view_2133, expand_22, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2130 = view_2133 = expand_22 = None
        getitem_812: "bf16[1024, 256, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
        convert_element_type_117: "f32[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_812, torch.float32);  getitem_812 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_342: "i8[100352, 256]" = torch.ops.aten.full.default([100352, 256], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_366 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 677, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_538, 'Y_ptr': full_default_342, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_538 = None
        getitem_814: "i8[100352, 256]" = triton_kernel_wrapper_functional_proxy_366['Y_ptr'];  triton_kernel_wrapper_functional_proxy_366 = None
        view_2137: "i8[512, 196, 256]" = torch.ops.aten.view.default(getitem_814, [512, 196, 256]);  getitem_814 = None
        view_2138: "i8[512, 50176]" = torch.ops.aten.view.default(view_2137, [512, -1]);  view_2137 = None
        view_2139: "i8[512, 256, 14, 14]" = torch.ops.aten.view.default(view_2138, [512, 256, 14, 14]);  view_2138 = None
        mul_381: "bf16[512, 256, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_808, view_2139);  getitem_808 = view_2139 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_532: "bf16[100352, 256]" = torch.ops.aten.empty.memory_format([100352, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_367 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 480, constant_args_idx = 678, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_533, 'S_ptr': getitem_534, 'M_ptr': getitem_535, 'Y_ptr': empty_532, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_533 = getitem_534 = getitem_535 = empty_532 = None
        getitem_815: "bf16[100352, 256]" = triton_kernel_wrapper_functional_proxy_367['Y_ptr'];  triton_kernel_wrapper_functional_proxy_367 = None
        view_2155: "bf16[512, 256, 196]" = torch.ops.aten.view.default(mul_381, [512, 256, 196]);  mul_381 = None
        view_2156: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_815, [512, 196, 256]);  getitem_815 = None
        view_2157: "bf16[512, 50176]" = torch.ops.aten.view.default(view_2156, [512, -1]);  view_2156 = None
        view_2158: "bf16[512, 256, 196]" = torch.ops.aten.view.default(view_2157, [512, 256, 196]);  view_2157 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        full_default_18: "f32[256]" = torch.ops.aten.full.default([256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        triton_kernel_wrapper_functional_proxy_368 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 481, constant_args_idx = 679, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2158, 'DY': view_2155, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_816: "f32[256]" = triton_kernel_wrapper_functional_proxy_368['DBETA']
        getitem_817: "f32[256]" = triton_kernel_wrapper_functional_proxy_368['DGAMMA'];  triton_kernel_wrapper_functional_proxy_368 = None
        empty_533: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_168: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_533, [0, 1, 2]);  empty_533 = None
        triton_kernel_wrapper_functional_proxy_369 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 482, constant_args_idx = 680, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2158, 'DY': view_2155, 'INVSTD': rsqrt_41, 'GAMMA': primals_250, 'DBETA': getitem_816, 'DGAMMA': getitem_817, 'DX': permute_168, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2158 = view_2155 = rsqrt_41 = primals_250 = permute_168 = None
        getitem_818: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_369['DX'];  triton_kernel_wrapper_functional_proxy_369 = None
        convert_element_type_default_83: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_817, torch.float32);  getitem_817 = None
        convert_element_type_default_82: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_816, torch.float32);  getitem_816 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_534: "bf16[100352, 256]" = torch.ops.aten.empty.memory_format([100352, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_370 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 483, constant_args_idx = 681, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_526, 'S_ptr': getitem_527, 'M_ptr': getitem_528, 'Y_ptr': empty_534, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_526 = getitem_527 = getitem_528 = empty_534 = None
        getitem_819: "bf16[100352, 256]" = triton_kernel_wrapper_functional_proxy_370['Y_ptr'];  triton_kernel_wrapper_functional_proxy_370 = None
        view_2175: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(getitem_818, [512, 256, 14, 14]);  getitem_818 = None
        empty_535: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_23: "bf16[512, 256, 14, 14]" = torch.ops.aten.expand.default(empty_535, [512, 256, 14, 14]);  empty_535 = None
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(view_2175, expand_23, convert_element_type_42, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_23 = convert_element_type_42 = None
        getitem_820: "bf16[512, 256, 14, 14]" = convolution_backward_22[0];  convolution_backward_22 = None
        empty_536: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_24: "bf16[256, 256, 3, 3]" = torch.ops.aten.expand.default(empty_536, [256, 256, 3, 3]);  empty_536 = None
        view_2176: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_819, [512, 196, 256]);  getitem_819 = None
        view_2177: "bf16[512, 50176]" = torch.ops.aten.view.default(view_2176, [512, -1]);  view_2176 = None
        view_2178: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(view_2177, [512, 256, 14, 14]);  view_2177 = None
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(view_2175, view_2178, expand_24, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_2175 = view_2178 = expand_24 = None
        getitem_824: "bf16[256, 256, 3, 3]" = convolution_backward_23[1];  convolution_backward_23 = None
        convert_element_type_122: "f32[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_824, torch.float32);  getitem_824 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_371 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 682, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_525, 'Y_ptr': full_default_342, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_525 = None
        getitem_826: "i8[100352, 256]" = triton_kernel_wrapper_functional_proxy_371['Y_ptr'];  triton_kernel_wrapper_functional_proxy_371 = None
        view_2182: "i8[512, 196, 256]" = torch.ops.aten.view.default(getitem_826, [512, 196, 256]);  getitem_826 = None
        view_2183: "i8[512, 50176]" = torch.ops.aten.view.default(view_2182, [512, -1]);  view_2182 = None
        view_2184: "i8[512, 256, 14, 14]" = torch.ops.aten.view.default(view_2183, [512, 256, 14, 14]);  view_2183 = None
        mul_382: "bf16[512, 256, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_820, view_2184);  getitem_820 = view_2184 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_537: "bf16[100352, 256]" = torch.ops.aten.empty.memory_format([100352, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_372 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 484, constant_args_idx = 683, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_520, 'S_ptr': getitem_521, 'M_ptr': getitem_522, 'Y_ptr': empty_537, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_520 = getitem_521 = getitem_522 = empty_537 = None
        getitem_827: "bf16[100352, 256]" = triton_kernel_wrapper_functional_proxy_372['Y_ptr'];  triton_kernel_wrapper_functional_proxy_372 = None
        view_2200: "bf16[512, 256, 196]" = torch.ops.aten.view.default(mul_382, [512, 256, 196]);  mul_382 = None
        view_2201: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_827, [512, 196, 256]);  getitem_827 = None
        view_2202: "bf16[512, 50176]" = torch.ops.aten.view.default(view_2201, [512, -1]);  view_2201 = None
        view_2203: "bf16[512, 256, 196]" = torch.ops.aten.view.default(view_2202, [512, 256, 196]);  view_2202 = None
        triton_kernel_wrapper_functional_proxy_373 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 485, constant_args_idx = 684, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2203, 'DY': view_2200, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_828: "f32[256]" = triton_kernel_wrapper_functional_proxy_373['DBETA']
        getitem_829: "f32[256]" = triton_kernel_wrapper_functional_proxy_373['DGAMMA'];  triton_kernel_wrapper_functional_proxy_373 = None
        empty_538: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_169: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_538, [0, 1, 2]);  empty_538 = None
        triton_kernel_wrapper_functional_proxy_374 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 486, constant_args_idx = 685, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2203, 'DY': view_2200, 'INVSTD': rsqrt_40, 'GAMMA': primals_244, 'DBETA': getitem_828, 'DGAMMA': getitem_829, 'DX': permute_169, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2203 = view_2200 = rsqrt_40 = primals_244 = permute_169 = None
        getitem_830: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_374['DX'];  triton_kernel_wrapper_functional_proxy_374 = None
        convert_element_type_default_81: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_829, torch.float32);  getitem_829 = None
        convert_element_type_default_80: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_828, torch.float32);  getitem_828 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_539: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_375 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 487, constant_args_idx = 686, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_513, 'S_ptr': getitem_514, 'M_ptr': getitem_515, 'Y_ptr': empty_539, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_513 = getitem_514 = getitem_515 = empty_539 = None
        getitem_831: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_375['Y_ptr'];  triton_kernel_wrapper_functional_proxy_375 = None
        view_2220: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(getitem_830, [512, 256, 14, 14]);  getitem_830 = None
        empty_540: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_25: "bf16[512, 1024, 14, 14]" = torch.ops.aten.expand.default(empty_540, [512, 1024, 14, 14]);  empty_540 = None
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(view_2220, expand_25, convert_element_type_41, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_25 = convert_element_type_41 = None
        getitem_832: "bf16[512, 1024, 14, 14]" = convolution_backward_24[0];  convolution_backward_24 = None
        empty_541: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_26: "bf16[256, 1024, 1, 1]" = torch.ops.aten.expand.default(empty_541, [256, 1024, 1, 1]);  empty_541 = None
        view_2221: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_831, [512, 784, 256]);  getitem_831 = None
        view_2222: "bf16[512, 200704]" = torch.ops.aten.view.default(view_2221, [512, -1]);  view_2221 = None
        view_2223: "bf16[512, 1024, 14, 14]" = torch.ops.aten.view.default(view_2222, [512, 1024, 14, 14]);  view_2222 = None
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(view_2220, view_2223, expand_26, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2220 = view_2223 = expand_26 = None
        getitem_836: "bf16[256, 1024, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
        convert_element_type_127: "f32[256, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_836, torch.float32);  getitem_836 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_231: "bf16[512, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_380, getitem_832);  mul_380 = getitem_832 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_376 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 687, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_512, 'Y_ptr': full_default_339, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_512 = None
        getitem_838: "i8[401408, 256]" = triton_kernel_wrapper_functional_proxy_376['Y_ptr'];  triton_kernel_wrapper_functional_proxy_376 = None
        view_2227: "i8[512, 784, 256]" = torch.ops.aten.view.default(getitem_838, [512, 784, 256]);  getitem_838 = None
        view_2228: "i8[512, 200704]" = torch.ops.aten.view.default(view_2227, [512, -1]);  view_2227 = None
        view_2229: "i8[512, 1024, 14, 14]" = torch.ops.aten.view.default(view_2228, [512, 1024, 14, 14]);  view_2228 = None
        mul_383: "bf16[512, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(add_231, view_2229);  add_231 = view_2229 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_542: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_377 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 488, constant_args_idx = 688, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_507, 'S_ptr': getitem_508, 'M_ptr': getitem_509, 'Y_ptr': empty_542, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_507 = getitem_508 = getitem_509 = empty_542 = None
        getitem_839: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_377['Y_ptr'];  triton_kernel_wrapper_functional_proxy_377 = None
        view_2245: "bf16[512, 1024, 196]" = torch.ops.aten.view.default(mul_383, [512, 1024, 196])
        view_2246: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_839, [512, 784, 256]);  getitem_839 = None
        view_2247: "bf16[512, 200704]" = torch.ops.aten.view.default(view_2246, [512, -1]);  view_2246 = None
        view_2248: "bf16[512, 1024, 196]" = torch.ops.aten.view.default(view_2247, [512, 1024, 196]);  view_2247 = None
        triton_kernel_wrapper_functional_proxy_378 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 489, constant_args_idx = 689, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2248, 'DY': view_2245, 'DBETA': full_default_152, 'DGAMMA': full_default_152, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_840: "f32[1024]" = triton_kernel_wrapper_functional_proxy_378['DBETA']
        getitem_841: "f32[1024]" = triton_kernel_wrapper_functional_proxy_378['DGAMMA'];  triton_kernel_wrapper_functional_proxy_378 = None
        empty_543: "bf16[512, 1024, 196]" = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_170: "bf16[512, 1024, 196]" = torch.ops.aten.permute.default(empty_543, [0, 1, 2]);  empty_543 = None
        triton_kernel_wrapper_functional_proxy_379 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 490, constant_args_idx = 690, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2248, 'DY': view_2245, 'INVSTD': rsqrt_39, 'GAMMA': primals_238, 'DBETA': getitem_840, 'DGAMMA': getitem_841, 'DX': permute_170, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2248 = view_2245 = rsqrt_39 = primals_238 = permute_170 = None
        getitem_842: "bf16[512, 1024, 196]" = triton_kernel_wrapper_functional_proxy_379['DX'];  triton_kernel_wrapper_functional_proxy_379 = None
        convert_element_type_default_79: "f32[1024]" = torch.ops.prims.convert_element_type.default(getitem_841, torch.float32);  getitem_841 = None
        convert_element_type_default_78: "f32[1024]" = torch.ops.prims.convert_element_type.default(getitem_840, torch.float32);  getitem_840 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_544: "bf16[100352, 256]" = torch.ops.aten.empty.memory_format([100352, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_380 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 491, constant_args_idx = 691, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_500, 'S_ptr': getitem_501, 'M_ptr': getitem_502, 'Y_ptr': empty_544, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_500 = getitem_501 = getitem_502 = empty_544 = None
        getitem_843: "bf16[100352, 256]" = triton_kernel_wrapper_functional_proxy_380['Y_ptr'];  triton_kernel_wrapper_functional_proxy_380 = None
        view_2265: "bf16[512, 1024, 14, 14]" = torch.ops.aten.view.default(getitem_842, [512, 1024, 14, 14]);  getitem_842 = None
        empty_545: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_27: "bf16[512, 256, 14, 14]" = torch.ops.aten.expand.default(empty_545, [512, 256, 14, 14]);  empty_545 = None
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(view_2265, expand_27, convert_element_type_40, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_27 = convert_element_type_40 = None
        getitem_844: "bf16[512, 256, 14, 14]" = convolution_backward_26[0];  convolution_backward_26 = None
        empty_546: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_28: "bf16[1024, 256, 1, 1]" = torch.ops.aten.expand.default(empty_546, [1024, 256, 1, 1]);  empty_546 = None
        view_2266: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_843, [512, 196, 256]);  getitem_843 = None
        view_2267: "bf16[512, 50176]" = torch.ops.aten.view.default(view_2266, [512, -1]);  view_2266 = None
        view_2268: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(view_2267, [512, 256, 14, 14]);  view_2267 = None
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(view_2265, view_2268, expand_28, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2265 = view_2268 = expand_28 = None
        getitem_848: "bf16[1024, 256, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
        convert_element_type_132: "f32[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_848, torch.float32);  getitem_848 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_381 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 692, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_499, 'Y_ptr': full_default_342, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_499 = None
        getitem_850: "i8[100352, 256]" = triton_kernel_wrapper_functional_proxy_381['Y_ptr'];  triton_kernel_wrapper_functional_proxy_381 = None
        view_2272: "i8[512, 196, 256]" = torch.ops.aten.view.default(getitem_850, [512, 196, 256]);  getitem_850 = None
        view_2273: "i8[512, 50176]" = torch.ops.aten.view.default(view_2272, [512, -1]);  view_2272 = None
        view_2274: "i8[512, 256, 14, 14]" = torch.ops.aten.view.default(view_2273, [512, 256, 14, 14]);  view_2273 = None
        mul_384: "bf16[512, 256, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_844, view_2274);  getitem_844 = view_2274 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_547: "bf16[100352, 256]" = torch.ops.aten.empty.memory_format([100352, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_382 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 492, constant_args_idx = 693, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_494, 'S_ptr': getitem_495, 'M_ptr': getitem_496, 'Y_ptr': empty_547, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_494 = getitem_495 = getitem_496 = empty_547 = None
        getitem_851: "bf16[100352, 256]" = triton_kernel_wrapper_functional_proxy_382['Y_ptr'];  triton_kernel_wrapper_functional_proxy_382 = None
        view_2290: "bf16[512, 256, 196]" = torch.ops.aten.view.default(mul_384, [512, 256, 196]);  mul_384 = None
        view_2291: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_851, [512, 196, 256]);  getitem_851 = None
        view_2292: "bf16[512, 50176]" = torch.ops.aten.view.default(view_2291, [512, -1]);  view_2291 = None
        view_2293: "bf16[512, 256, 196]" = torch.ops.aten.view.default(view_2292, [512, 256, 196]);  view_2292 = None
        triton_kernel_wrapper_functional_proxy_383 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 493, constant_args_idx = 694, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2293, 'DY': view_2290, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_852: "f32[256]" = triton_kernel_wrapper_functional_proxy_383['DBETA']
        getitem_853: "f32[256]" = triton_kernel_wrapper_functional_proxy_383['DGAMMA'];  triton_kernel_wrapper_functional_proxy_383 = None
        empty_548: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_171: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_548, [0, 1, 2]);  empty_548 = None
        triton_kernel_wrapper_functional_proxy_384 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 494, constant_args_idx = 695, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2293, 'DY': view_2290, 'INVSTD': rsqrt_38, 'GAMMA': primals_232, 'DBETA': getitem_852, 'DGAMMA': getitem_853, 'DX': permute_171, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2293 = view_2290 = rsqrt_38 = primals_232 = permute_171 = None
        getitem_854: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_384['DX'];  triton_kernel_wrapper_functional_proxy_384 = None
        convert_element_type_default_77: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_853, torch.float32);  getitem_853 = None
        convert_element_type_default_76: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_852, torch.float32);  getitem_852 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_549: "bf16[100352, 256]" = torch.ops.aten.empty.memory_format([100352, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_385 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 495, constant_args_idx = 696, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_487, 'S_ptr': getitem_488, 'M_ptr': getitem_489, 'Y_ptr': empty_549, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_487 = getitem_488 = getitem_489 = empty_549 = None
        getitem_855: "bf16[100352, 256]" = triton_kernel_wrapper_functional_proxy_385['Y_ptr'];  triton_kernel_wrapper_functional_proxy_385 = None
        view_2310: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(getitem_854, [512, 256, 14, 14]);  getitem_854 = None
        empty_550: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_29: "bf16[512, 256, 14, 14]" = torch.ops.aten.expand.default(empty_550, [512, 256, 14, 14]);  empty_550 = None
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(view_2310, expand_29, convert_element_type_39, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_29 = convert_element_type_39 = None
        getitem_856: "bf16[512, 256, 14, 14]" = convolution_backward_28[0];  convolution_backward_28 = None
        empty_551: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_30: "bf16[256, 256, 3, 3]" = torch.ops.aten.expand.default(empty_551, [256, 256, 3, 3]);  empty_551 = None
        view_2311: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_855, [512, 196, 256]);  getitem_855 = None
        view_2312: "bf16[512, 50176]" = torch.ops.aten.view.default(view_2311, [512, -1]);  view_2311 = None
        view_2313: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(view_2312, [512, 256, 14, 14]);  view_2312 = None
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(view_2310, view_2313, expand_30, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_2310 = view_2313 = expand_30 = None
        getitem_860: "bf16[256, 256, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None
        convert_element_type_137: "f32[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_860, torch.float32);  getitem_860 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_386 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 697, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_486, 'Y_ptr': full_default_342, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_486 = None
        getitem_862: "i8[100352, 256]" = triton_kernel_wrapper_functional_proxy_386['Y_ptr'];  triton_kernel_wrapper_functional_proxy_386 = None
        view_2317: "i8[512, 196, 256]" = torch.ops.aten.view.default(getitem_862, [512, 196, 256]);  getitem_862 = None
        view_2318: "i8[512, 50176]" = torch.ops.aten.view.default(view_2317, [512, -1]);  view_2317 = None
        view_2319: "i8[512, 256, 14, 14]" = torch.ops.aten.view.default(view_2318, [512, 256, 14, 14]);  view_2318 = None
        mul_385: "bf16[512, 256, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_856, view_2319);  getitem_856 = view_2319 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_552: "bf16[100352, 256]" = torch.ops.aten.empty.memory_format([100352, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_387 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 496, constant_args_idx = 698, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_481, 'S_ptr': getitem_482, 'M_ptr': getitem_483, 'Y_ptr': empty_552, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_481 = getitem_482 = getitem_483 = empty_552 = None
        getitem_863: "bf16[100352, 256]" = triton_kernel_wrapper_functional_proxy_387['Y_ptr'];  triton_kernel_wrapper_functional_proxy_387 = None
        view_2335: "bf16[512, 256, 196]" = torch.ops.aten.view.default(mul_385, [512, 256, 196]);  mul_385 = None
        view_2336: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_863, [512, 196, 256]);  getitem_863 = None
        view_2337: "bf16[512, 50176]" = torch.ops.aten.view.default(view_2336, [512, -1]);  view_2336 = None
        view_2338: "bf16[512, 256, 196]" = torch.ops.aten.view.default(view_2337, [512, 256, 196]);  view_2337 = None
        triton_kernel_wrapper_functional_proxy_388 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 497, constant_args_idx = 699, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2338, 'DY': view_2335, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_864: "f32[256]" = triton_kernel_wrapper_functional_proxy_388['DBETA']
        getitem_865: "f32[256]" = triton_kernel_wrapper_functional_proxy_388['DGAMMA'];  triton_kernel_wrapper_functional_proxy_388 = None
        empty_553: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_172: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_553, [0, 1, 2]);  empty_553 = None
        triton_kernel_wrapper_functional_proxy_389 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 498, constant_args_idx = 700, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2338, 'DY': view_2335, 'INVSTD': rsqrt_37, 'GAMMA': primals_226, 'DBETA': getitem_864, 'DGAMMA': getitem_865, 'DX': permute_172, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2338 = view_2335 = rsqrt_37 = primals_226 = permute_172 = None
        getitem_866: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_389['DX'];  triton_kernel_wrapper_functional_proxy_389 = None
        convert_element_type_default_75: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_865, torch.float32);  getitem_865 = None
        convert_element_type_default_74: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_864, torch.float32);  getitem_864 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_554: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_390 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 499, constant_args_idx = 701, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_474, 'S_ptr': getitem_475, 'M_ptr': getitem_476, 'Y_ptr': empty_554, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_474 = getitem_475 = getitem_476 = empty_554 = None
        getitem_867: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_390['Y_ptr'];  triton_kernel_wrapper_functional_proxy_390 = None
        view_2355: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(getitem_866, [512, 256, 14, 14]);  getitem_866 = None
        empty_555: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_31: "bf16[512, 1024, 14, 14]" = torch.ops.aten.expand.default(empty_555, [512, 1024, 14, 14]);  empty_555 = None
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(view_2355, expand_31, convert_element_type_38, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_31 = convert_element_type_38 = None
        getitem_868: "bf16[512, 1024, 14, 14]" = convolution_backward_30[0];  convolution_backward_30 = None
        empty_556: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_32: "bf16[256, 1024, 1, 1]" = torch.ops.aten.expand.default(empty_556, [256, 1024, 1, 1]);  empty_556 = None
        view_2356: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_867, [512, 784, 256]);  getitem_867 = None
        view_2357: "bf16[512, 200704]" = torch.ops.aten.view.default(view_2356, [512, -1]);  view_2356 = None
        view_2358: "bf16[512, 1024, 14, 14]" = torch.ops.aten.view.default(view_2357, [512, 1024, 14, 14]);  view_2357 = None
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(view_2355, view_2358, expand_32, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2355 = view_2358 = expand_32 = None
        getitem_872: "bf16[256, 1024, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
        convert_element_type_142: "f32[256, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_872, torch.float32);  getitem_872 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_232: "bf16[512, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_383, getitem_868);  mul_383 = getitem_868 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_391 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 702, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_473, 'Y_ptr': full_default_339, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_473 = None
        getitem_874: "i8[401408, 256]" = triton_kernel_wrapper_functional_proxy_391['Y_ptr'];  triton_kernel_wrapper_functional_proxy_391 = None
        view_2362: "i8[512, 784, 256]" = torch.ops.aten.view.default(getitem_874, [512, 784, 256]);  getitem_874 = None
        view_2363: "i8[512, 200704]" = torch.ops.aten.view.default(view_2362, [512, -1]);  view_2362 = None
        view_2364: "i8[512, 1024, 14, 14]" = torch.ops.aten.view.default(view_2363, [512, 1024, 14, 14]);  view_2363 = None
        mul_386: "bf16[512, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(add_232, view_2364);  add_232 = view_2364 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_557: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_392 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 500, constant_args_idx = 703, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_468, 'S_ptr': getitem_469, 'M_ptr': getitem_470, 'Y_ptr': empty_557, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_468 = getitem_469 = getitem_470 = empty_557 = None
        getitem_875: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_392['Y_ptr'];  triton_kernel_wrapper_functional_proxy_392 = None
        view_2380: "bf16[512, 1024, 196]" = torch.ops.aten.view.default(mul_386, [512, 1024, 196])
        view_2381: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_875, [512, 784, 256]);  getitem_875 = None
        view_2382: "bf16[512, 200704]" = torch.ops.aten.view.default(view_2381, [512, -1]);  view_2381 = None
        view_2383: "bf16[512, 1024, 196]" = torch.ops.aten.view.default(view_2382, [512, 1024, 196]);  view_2382 = None
        triton_kernel_wrapper_functional_proxy_393 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 501, constant_args_idx = 704, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2383, 'DY': view_2380, 'DBETA': full_default_152, 'DGAMMA': full_default_152, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_876: "f32[1024]" = triton_kernel_wrapper_functional_proxy_393['DBETA']
        getitem_877: "f32[1024]" = triton_kernel_wrapper_functional_proxy_393['DGAMMA'];  triton_kernel_wrapper_functional_proxy_393 = None
        empty_558: "bf16[512, 1024, 196]" = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_173: "bf16[512, 1024, 196]" = torch.ops.aten.permute.default(empty_558, [0, 1, 2]);  empty_558 = None
        triton_kernel_wrapper_functional_proxy_394 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 502, constant_args_idx = 705, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2383, 'DY': view_2380, 'INVSTD': rsqrt_36, 'GAMMA': primals_220, 'DBETA': getitem_876, 'DGAMMA': getitem_877, 'DX': permute_173, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2383 = view_2380 = rsqrt_36 = primals_220 = permute_173 = None
        getitem_878: "bf16[512, 1024, 196]" = triton_kernel_wrapper_functional_proxy_394['DX'];  triton_kernel_wrapper_functional_proxy_394 = None
        convert_element_type_default_73: "f32[1024]" = torch.ops.prims.convert_element_type.default(getitem_877, torch.float32);  getitem_877 = None
        convert_element_type_default_72: "f32[1024]" = torch.ops.prims.convert_element_type.default(getitem_876, torch.float32);  getitem_876 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_559: "bf16[100352, 256]" = torch.ops.aten.empty.memory_format([100352, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_395 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 503, constant_args_idx = 706, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_461, 'S_ptr': getitem_462, 'M_ptr': getitem_463, 'Y_ptr': empty_559, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_461 = getitem_462 = getitem_463 = empty_559 = None
        getitem_879: "bf16[100352, 256]" = triton_kernel_wrapper_functional_proxy_395['Y_ptr'];  triton_kernel_wrapper_functional_proxy_395 = None
        view_2400: "bf16[512, 1024, 14, 14]" = torch.ops.aten.view.default(getitem_878, [512, 1024, 14, 14]);  getitem_878 = None
        empty_560: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_33: "bf16[512, 256, 14, 14]" = torch.ops.aten.expand.default(empty_560, [512, 256, 14, 14]);  empty_560 = None
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(view_2400, expand_33, convert_element_type_37, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_33 = convert_element_type_37 = None
        getitem_880: "bf16[512, 256, 14, 14]" = convolution_backward_32[0];  convolution_backward_32 = None
        empty_561: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_34: "bf16[1024, 256, 1, 1]" = torch.ops.aten.expand.default(empty_561, [1024, 256, 1, 1]);  empty_561 = None
        view_2401: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_879, [512, 196, 256]);  getitem_879 = None
        view_2402: "bf16[512, 50176]" = torch.ops.aten.view.default(view_2401, [512, -1]);  view_2401 = None
        view_2403: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(view_2402, [512, 256, 14, 14]);  view_2402 = None
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(view_2400, view_2403, expand_34, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2400 = view_2403 = expand_34 = None
        getitem_884: "bf16[1024, 256, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
        convert_element_type_147: "f32[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_884, torch.float32);  getitem_884 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_396 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 707, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_460, 'Y_ptr': full_default_342, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_460 = None
        getitem_886: "i8[100352, 256]" = triton_kernel_wrapper_functional_proxy_396['Y_ptr'];  triton_kernel_wrapper_functional_proxy_396 = None
        view_2407: "i8[512, 196, 256]" = torch.ops.aten.view.default(getitem_886, [512, 196, 256]);  getitem_886 = None
        view_2408: "i8[512, 50176]" = torch.ops.aten.view.default(view_2407, [512, -1]);  view_2407 = None
        view_2409: "i8[512, 256, 14, 14]" = torch.ops.aten.view.default(view_2408, [512, 256, 14, 14]);  view_2408 = None
        mul_387: "bf16[512, 256, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_880, view_2409);  getitem_880 = view_2409 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_562: "bf16[100352, 256]" = torch.ops.aten.empty.memory_format([100352, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_397 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 504, constant_args_idx = 708, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_455, 'S_ptr': getitem_456, 'M_ptr': getitem_457, 'Y_ptr': empty_562, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_455 = getitem_456 = getitem_457 = empty_562 = None
        getitem_887: "bf16[100352, 256]" = triton_kernel_wrapper_functional_proxy_397['Y_ptr'];  triton_kernel_wrapper_functional_proxy_397 = None
        view_2425: "bf16[512, 256, 196]" = torch.ops.aten.view.default(mul_387, [512, 256, 196]);  mul_387 = None
        view_2426: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_887, [512, 196, 256]);  getitem_887 = None
        view_2427: "bf16[512, 50176]" = torch.ops.aten.view.default(view_2426, [512, -1]);  view_2426 = None
        view_2428: "bf16[512, 256, 196]" = torch.ops.aten.view.default(view_2427, [512, 256, 196]);  view_2427 = None
        triton_kernel_wrapper_functional_proxy_398 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 505, constant_args_idx = 709, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2428, 'DY': view_2425, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_888: "f32[256]" = triton_kernel_wrapper_functional_proxy_398['DBETA']
        getitem_889: "f32[256]" = triton_kernel_wrapper_functional_proxy_398['DGAMMA'];  triton_kernel_wrapper_functional_proxy_398 = None
        empty_563: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_174: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_563, [0, 1, 2]);  empty_563 = None
        triton_kernel_wrapper_functional_proxy_399 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 506, constant_args_idx = 710, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2428, 'DY': view_2425, 'INVSTD': rsqrt_35, 'GAMMA': primals_214, 'DBETA': getitem_888, 'DGAMMA': getitem_889, 'DX': permute_174, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2428 = view_2425 = rsqrt_35 = primals_214 = permute_174 = None
        getitem_890: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_399['DX'];  triton_kernel_wrapper_functional_proxy_399 = None
        convert_element_type_default_71: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_889, torch.float32);  getitem_889 = None
        convert_element_type_default_70: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_888, torch.float32);  getitem_888 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_564: "bf16[100352, 256]" = torch.ops.aten.empty.memory_format([100352, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_400 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 507, constant_args_idx = 711, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_448, 'S_ptr': getitem_449, 'M_ptr': getitem_450, 'Y_ptr': empty_564, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_448 = getitem_449 = getitem_450 = empty_564 = None
        getitem_891: "bf16[100352, 256]" = triton_kernel_wrapper_functional_proxy_400['Y_ptr'];  triton_kernel_wrapper_functional_proxy_400 = None
        view_2445: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(getitem_890, [512, 256, 14, 14]);  getitem_890 = None
        empty_565: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_35: "bf16[512, 256, 14, 14]" = torch.ops.aten.expand.default(empty_565, [512, 256, 14, 14]);  empty_565 = None
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(view_2445, expand_35, convert_element_type_36, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_35 = convert_element_type_36 = None
        getitem_892: "bf16[512, 256, 14, 14]" = convolution_backward_34[0];  convolution_backward_34 = None
        empty_566: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_36: "bf16[256, 256, 3, 3]" = torch.ops.aten.expand.default(empty_566, [256, 256, 3, 3]);  empty_566 = None
        view_2446: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_891, [512, 196, 256]);  getitem_891 = None
        view_2447: "bf16[512, 50176]" = torch.ops.aten.view.default(view_2446, [512, -1]);  view_2446 = None
        view_2448: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(view_2447, [512, 256, 14, 14]);  view_2447 = None
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(view_2445, view_2448, expand_36, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_2445 = view_2448 = expand_36 = None
        getitem_896: "bf16[256, 256, 3, 3]" = convolution_backward_35[1];  convolution_backward_35 = None
        convert_element_type_152: "f32[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_896, torch.float32);  getitem_896 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_401 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 712, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_447, 'Y_ptr': full_default_342, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_447 = None
        getitem_898: "i8[100352, 256]" = triton_kernel_wrapper_functional_proxy_401['Y_ptr'];  triton_kernel_wrapper_functional_proxy_401 = None
        view_2452: "i8[512, 196, 256]" = torch.ops.aten.view.default(getitem_898, [512, 196, 256]);  getitem_898 = None
        view_2453: "i8[512, 50176]" = torch.ops.aten.view.default(view_2452, [512, -1]);  view_2452 = None
        view_2454: "i8[512, 256, 14, 14]" = torch.ops.aten.view.default(view_2453, [512, 256, 14, 14]);  view_2453 = None
        mul_388: "bf16[512, 256, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_892, view_2454);  getitem_892 = view_2454 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_567: "bf16[100352, 256]" = torch.ops.aten.empty.memory_format([100352, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_402 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 508, constant_args_idx = 713, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_442, 'S_ptr': getitem_443, 'M_ptr': getitem_444, 'Y_ptr': empty_567, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_442 = getitem_443 = getitem_444 = empty_567 = None
        getitem_899: "bf16[100352, 256]" = triton_kernel_wrapper_functional_proxy_402['Y_ptr'];  triton_kernel_wrapper_functional_proxy_402 = None
        view_2470: "bf16[512, 256, 196]" = torch.ops.aten.view.default(mul_388, [512, 256, 196]);  mul_388 = None
        view_2471: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_899, [512, 196, 256]);  getitem_899 = None
        view_2472: "bf16[512, 50176]" = torch.ops.aten.view.default(view_2471, [512, -1]);  view_2471 = None
        view_2473: "bf16[512, 256, 196]" = torch.ops.aten.view.default(view_2472, [512, 256, 196]);  view_2472 = None
        triton_kernel_wrapper_functional_proxy_403 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 509, constant_args_idx = 714, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2473, 'DY': view_2470, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_900: "f32[256]" = triton_kernel_wrapper_functional_proxy_403['DBETA']
        getitem_901: "f32[256]" = triton_kernel_wrapper_functional_proxy_403['DGAMMA'];  triton_kernel_wrapper_functional_proxy_403 = None
        empty_568: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_175: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_568, [0, 1, 2]);  empty_568 = None
        triton_kernel_wrapper_functional_proxy_404 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 510, constant_args_idx = 715, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2473, 'DY': view_2470, 'INVSTD': rsqrt_34, 'GAMMA': primals_208, 'DBETA': getitem_900, 'DGAMMA': getitem_901, 'DX': permute_175, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2473 = view_2470 = rsqrt_34 = primals_208 = permute_175 = None
        getitem_902: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_404['DX'];  triton_kernel_wrapper_functional_proxy_404 = None
        convert_element_type_default_69: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_901, torch.float32);  getitem_901 = None
        convert_element_type_default_68: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_900, torch.float32);  getitem_900 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_569: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_405 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 511, constant_args_idx = 716, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_435, 'S_ptr': getitem_436, 'M_ptr': getitem_437, 'Y_ptr': empty_569, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_435 = getitem_436 = getitem_437 = empty_569 = None
        getitem_903: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_405['Y_ptr'];  triton_kernel_wrapper_functional_proxy_405 = None
        view_2490: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(getitem_902, [512, 256, 14, 14]);  getitem_902 = None
        empty_570: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_37: "bf16[512, 1024, 14, 14]" = torch.ops.aten.expand.default(empty_570, [512, 1024, 14, 14]);  empty_570 = None
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(view_2490, expand_37, convert_element_type_35, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_37 = convert_element_type_35 = None
        getitem_904: "bf16[512, 1024, 14, 14]" = convolution_backward_36[0];  convolution_backward_36 = None
        empty_571: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_38: "bf16[256, 1024, 1, 1]" = torch.ops.aten.expand.default(empty_571, [256, 1024, 1, 1]);  empty_571 = None
        view_2491: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_903, [512, 784, 256]);  getitem_903 = None
        view_2492: "bf16[512, 200704]" = torch.ops.aten.view.default(view_2491, [512, -1]);  view_2491 = None
        view_2493: "bf16[512, 1024, 14, 14]" = torch.ops.aten.view.default(view_2492, [512, 1024, 14, 14]);  view_2492 = None
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(view_2490, view_2493, expand_38, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2490 = view_2493 = expand_38 = None
        getitem_908: "bf16[256, 1024, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
        convert_element_type_157: "f32[256, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_908, torch.float32);  getitem_908 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_233: "bf16[512, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_386, getitem_904);  mul_386 = getitem_904 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_406 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 717, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_434, 'Y_ptr': full_default_339, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_434 = None
        getitem_910: "i8[401408, 256]" = triton_kernel_wrapper_functional_proxy_406['Y_ptr'];  triton_kernel_wrapper_functional_proxy_406 = None
        view_2497: "i8[512, 784, 256]" = torch.ops.aten.view.default(getitem_910, [512, 784, 256]);  getitem_910 = None
        view_2498: "i8[512, 200704]" = torch.ops.aten.view.default(view_2497, [512, -1]);  view_2497 = None
        view_2499: "i8[512, 1024, 14, 14]" = torch.ops.aten.view.default(view_2498, [512, 1024, 14, 14]);  view_2498 = None
        mul_389: "bf16[512, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(add_233, view_2499);  add_233 = view_2499 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_572: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_407 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 512, constant_args_idx = 718, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_429, 'S_ptr': getitem_430, 'M_ptr': getitem_431, 'Y_ptr': empty_572, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_429 = getitem_430 = getitem_431 = empty_572 = None
        getitem_911: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_407['Y_ptr'];  triton_kernel_wrapper_functional_proxy_407 = None
        view_2515: "bf16[512, 1024, 196]" = torch.ops.aten.view.default(mul_389, [512, 1024, 196])
        view_2516: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_911, [512, 784, 256]);  getitem_911 = None
        view_2517: "bf16[512, 200704]" = torch.ops.aten.view.default(view_2516, [512, -1]);  view_2516 = None
        view_2518: "bf16[512, 1024, 196]" = torch.ops.aten.view.default(view_2517, [512, 1024, 196]);  view_2517 = None
        triton_kernel_wrapper_functional_proxy_408 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 513, constant_args_idx = 719, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2518, 'DY': view_2515, 'DBETA': full_default_152, 'DGAMMA': full_default_152, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_912: "f32[1024]" = triton_kernel_wrapper_functional_proxy_408['DBETA']
        getitem_913: "f32[1024]" = triton_kernel_wrapper_functional_proxy_408['DGAMMA'];  triton_kernel_wrapper_functional_proxy_408 = None
        empty_573: "bf16[512, 1024, 196]" = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_176: "bf16[512, 1024, 196]" = torch.ops.aten.permute.default(empty_573, [0, 1, 2]);  empty_573 = None
        triton_kernel_wrapper_functional_proxy_409 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 514, constant_args_idx = 720, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2518, 'DY': view_2515, 'INVSTD': rsqrt_33, 'GAMMA': primals_202, 'DBETA': getitem_912, 'DGAMMA': getitem_913, 'DX': permute_176, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2518 = view_2515 = rsqrt_33 = primals_202 = permute_176 = None
        getitem_914: "bf16[512, 1024, 196]" = triton_kernel_wrapper_functional_proxy_409['DX'];  triton_kernel_wrapper_functional_proxy_409 = None
        convert_element_type_default_67: "f32[1024]" = torch.ops.prims.convert_element_type.default(getitem_913, torch.float32);  getitem_913 = None
        convert_element_type_default_66: "f32[1024]" = torch.ops.prims.convert_element_type.default(getitem_912, torch.float32);  getitem_912 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_574: "bf16[100352, 256]" = torch.ops.aten.empty.memory_format([100352, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_410 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 515, constant_args_idx = 721, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_422, 'S_ptr': getitem_423, 'M_ptr': getitem_424, 'Y_ptr': empty_574, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_422 = getitem_423 = getitem_424 = empty_574 = None
        getitem_915: "bf16[100352, 256]" = triton_kernel_wrapper_functional_proxy_410['Y_ptr'];  triton_kernel_wrapper_functional_proxy_410 = None
        view_2535: "bf16[512, 1024, 14, 14]" = torch.ops.aten.view.default(getitem_914, [512, 1024, 14, 14]);  getitem_914 = None
        empty_575: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_39: "bf16[512, 256, 14, 14]" = torch.ops.aten.expand.default(empty_575, [512, 256, 14, 14]);  empty_575 = None
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(view_2535, expand_39, convert_element_type_34, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_39 = convert_element_type_34 = None
        getitem_916: "bf16[512, 256, 14, 14]" = convolution_backward_38[0];  convolution_backward_38 = None
        empty_576: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_40: "bf16[1024, 256, 1, 1]" = torch.ops.aten.expand.default(empty_576, [1024, 256, 1, 1]);  empty_576 = None
        view_2536: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_915, [512, 196, 256]);  getitem_915 = None
        view_2537: "bf16[512, 50176]" = torch.ops.aten.view.default(view_2536, [512, -1]);  view_2536 = None
        view_2538: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(view_2537, [512, 256, 14, 14]);  view_2537 = None
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(view_2535, view_2538, expand_40, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2535 = view_2538 = expand_40 = None
        getitem_920: "bf16[1024, 256, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None
        convert_element_type_162: "f32[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_920, torch.float32);  getitem_920 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_411 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 722, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_421, 'Y_ptr': full_default_342, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_421 = None
        getitem_922: "i8[100352, 256]" = triton_kernel_wrapper_functional_proxy_411['Y_ptr'];  triton_kernel_wrapper_functional_proxy_411 = None
        view_2542: "i8[512, 196, 256]" = torch.ops.aten.view.default(getitem_922, [512, 196, 256]);  getitem_922 = None
        view_2543: "i8[512, 50176]" = torch.ops.aten.view.default(view_2542, [512, -1]);  view_2542 = None
        view_2544: "i8[512, 256, 14, 14]" = torch.ops.aten.view.default(view_2543, [512, 256, 14, 14]);  view_2543 = None
        mul_390: "bf16[512, 256, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_916, view_2544);  getitem_916 = view_2544 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_577: "bf16[100352, 256]" = torch.ops.aten.empty.memory_format([100352, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_412 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 516, constant_args_idx = 723, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_416, 'S_ptr': getitem_417, 'M_ptr': getitem_418, 'Y_ptr': empty_577, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_416 = getitem_417 = getitem_418 = empty_577 = None
        getitem_923: "bf16[100352, 256]" = triton_kernel_wrapper_functional_proxy_412['Y_ptr'];  triton_kernel_wrapper_functional_proxy_412 = None
        view_2560: "bf16[512, 256, 196]" = torch.ops.aten.view.default(mul_390, [512, 256, 196]);  mul_390 = None
        view_2561: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_923, [512, 196, 256]);  getitem_923 = None
        view_2562: "bf16[512, 50176]" = torch.ops.aten.view.default(view_2561, [512, -1]);  view_2561 = None
        view_2563: "bf16[512, 256, 196]" = torch.ops.aten.view.default(view_2562, [512, 256, 196]);  view_2562 = None
        triton_kernel_wrapper_functional_proxy_413 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 517, constant_args_idx = 724, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2563, 'DY': view_2560, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_924: "f32[256]" = triton_kernel_wrapper_functional_proxy_413['DBETA']
        getitem_925: "f32[256]" = triton_kernel_wrapper_functional_proxy_413['DGAMMA'];  triton_kernel_wrapper_functional_proxy_413 = None
        empty_578: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_177: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_578, [0, 1, 2]);  empty_578 = None
        triton_kernel_wrapper_functional_proxy_414 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 518, constant_args_idx = 725, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2563, 'DY': view_2560, 'INVSTD': rsqrt_32, 'GAMMA': primals_196, 'DBETA': getitem_924, 'DGAMMA': getitem_925, 'DX': permute_177, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2563 = view_2560 = rsqrt_32 = primals_196 = permute_177 = None
        getitem_926: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_414['DX'];  triton_kernel_wrapper_functional_proxy_414 = None
        convert_element_type_default_65: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_925, torch.float32);  getitem_925 = None
        convert_element_type_default_64: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_924, torch.float32);  getitem_924 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_579: "bf16[100352, 256]" = torch.ops.aten.empty.memory_format([100352, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_415 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 519, constant_args_idx = 726, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_409, 'S_ptr': getitem_410, 'M_ptr': getitem_411, 'Y_ptr': empty_579, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_409 = getitem_410 = getitem_411 = empty_579 = None
        getitem_927: "bf16[100352, 256]" = triton_kernel_wrapper_functional_proxy_415['Y_ptr'];  triton_kernel_wrapper_functional_proxy_415 = None
        view_2580: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(getitem_926, [512, 256, 14, 14]);  getitem_926 = None
        empty_580: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_41: "bf16[512, 256, 14, 14]" = torch.ops.aten.expand.default(empty_580, [512, 256, 14, 14]);  empty_580 = None
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(view_2580, expand_41, convert_element_type_33, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_41 = convert_element_type_33 = None
        getitem_928: "bf16[512, 256, 14, 14]" = convolution_backward_40[0];  convolution_backward_40 = None
        empty_581: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_42: "bf16[256, 256, 3, 3]" = torch.ops.aten.expand.default(empty_581, [256, 256, 3, 3]);  empty_581 = None
        view_2581: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_927, [512, 196, 256]);  getitem_927 = None
        view_2582: "bf16[512, 50176]" = torch.ops.aten.view.default(view_2581, [512, -1]);  view_2581 = None
        view_2583: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(view_2582, [512, 256, 14, 14]);  view_2582 = None
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(view_2580, view_2583, expand_42, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_2580 = view_2583 = expand_42 = None
        getitem_932: "bf16[256, 256, 3, 3]" = convolution_backward_41[1];  convolution_backward_41 = None
        convert_element_type_167: "f32[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_932, torch.float32);  getitem_932 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_416 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 727, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_408, 'Y_ptr': full_default_342, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_408 = None
        getitem_934: "i8[100352, 256]" = triton_kernel_wrapper_functional_proxy_416['Y_ptr'];  triton_kernel_wrapper_functional_proxy_416 = None
        view_2587: "i8[512, 196, 256]" = torch.ops.aten.view.default(getitem_934, [512, 196, 256]);  getitem_934 = None
        view_2588: "i8[512, 50176]" = torch.ops.aten.view.default(view_2587, [512, -1]);  view_2587 = None
        view_2589: "i8[512, 256, 14, 14]" = torch.ops.aten.view.default(view_2588, [512, 256, 14, 14]);  view_2588 = None
        mul_391: "bf16[512, 256, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_928, view_2589);  getitem_928 = view_2589 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_582: "bf16[100352, 256]" = torch.ops.aten.empty.memory_format([100352, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_417 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 520, constant_args_idx = 728, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_403, 'S_ptr': getitem_404, 'M_ptr': getitem_405, 'Y_ptr': empty_582, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_403 = getitem_404 = getitem_405 = empty_582 = None
        getitem_935: "bf16[100352, 256]" = triton_kernel_wrapper_functional_proxy_417['Y_ptr'];  triton_kernel_wrapper_functional_proxy_417 = None
        view_2605: "bf16[512, 256, 196]" = torch.ops.aten.view.default(mul_391, [512, 256, 196]);  mul_391 = None
        view_2606: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_935, [512, 196, 256]);  getitem_935 = None
        view_2607: "bf16[512, 50176]" = torch.ops.aten.view.default(view_2606, [512, -1]);  view_2606 = None
        view_2608: "bf16[512, 256, 196]" = torch.ops.aten.view.default(view_2607, [512, 256, 196]);  view_2607 = None
        triton_kernel_wrapper_functional_proxy_418 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 521, constant_args_idx = 729, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2608, 'DY': view_2605, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_936: "f32[256]" = triton_kernel_wrapper_functional_proxy_418['DBETA']
        getitem_937: "f32[256]" = triton_kernel_wrapper_functional_proxy_418['DGAMMA'];  triton_kernel_wrapper_functional_proxy_418 = None
        empty_583: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_178: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_583, [0, 1, 2]);  empty_583 = None
        triton_kernel_wrapper_functional_proxy_419 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 522, constant_args_idx = 730, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2608, 'DY': view_2605, 'INVSTD': rsqrt_31, 'GAMMA': primals_190, 'DBETA': getitem_936, 'DGAMMA': getitem_937, 'DX': permute_178, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2608 = view_2605 = rsqrt_31 = primals_190 = permute_178 = None
        getitem_938: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_419['DX'];  triton_kernel_wrapper_functional_proxy_419 = None
        convert_element_type_default_63: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_937, torch.float32);  getitem_937 = None
        convert_element_type_default_62: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_936, torch.float32);  getitem_936 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_584: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_420 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 523, constant_args_idx = 731, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_396, 'S_ptr': getitem_397, 'M_ptr': getitem_398, 'Y_ptr': empty_584, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_396 = getitem_397 = getitem_398 = empty_584 = None
        getitem_939: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_420['Y_ptr'];  triton_kernel_wrapper_functional_proxy_420 = None
        view_2625: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(getitem_938, [512, 256, 14, 14]);  getitem_938 = None
        empty_585: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_43: "bf16[512, 1024, 14, 14]" = torch.ops.aten.expand.default(empty_585, [512, 1024, 14, 14]);  empty_585 = None
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(view_2625, expand_43, convert_element_type_32, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_43 = convert_element_type_32 = None
        getitem_940: "bf16[512, 1024, 14, 14]" = convolution_backward_42[0];  convolution_backward_42 = None
        empty_586: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_44: "bf16[256, 1024, 1, 1]" = torch.ops.aten.expand.default(empty_586, [256, 1024, 1, 1]);  empty_586 = None
        view_2626: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_939, [512, 784, 256]);  getitem_939 = None
        view_2627: "bf16[512, 200704]" = torch.ops.aten.view.default(view_2626, [512, -1]);  view_2626 = None
        view_2628: "bf16[512, 1024, 14, 14]" = torch.ops.aten.view.default(view_2627, [512, 1024, 14, 14]);  view_2627 = None
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(view_2625, view_2628, expand_44, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2625 = view_2628 = expand_44 = None
        getitem_944: "bf16[256, 1024, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
        convert_element_type_172: "f32[256, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_944, torch.float32);  getitem_944 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_234: "bf16[512, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_389, getitem_940);  mul_389 = getitem_940 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_421 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 732, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_395, 'Y_ptr': full_default_339, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_395 = None
        getitem_946: "i8[401408, 256]" = triton_kernel_wrapper_functional_proxy_421['Y_ptr'];  triton_kernel_wrapper_functional_proxy_421 = None
        view_2632: "i8[512, 784, 256]" = torch.ops.aten.view.default(getitem_946, [512, 784, 256]);  getitem_946 = None
        view_2633: "i8[512, 200704]" = torch.ops.aten.view.default(view_2632, [512, -1]);  view_2632 = None
        view_2634: "i8[512, 1024, 14, 14]" = torch.ops.aten.view.default(view_2633, [512, 1024, 14, 14]);  view_2633 = None
        mul_392: "bf16[512, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(add_234, view_2634);  add_234 = view_2634 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_587: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_422 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 524, constant_args_idx = 733, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_390, 'S_ptr': getitem_391, 'M_ptr': getitem_392, 'Y_ptr': empty_587, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_390 = getitem_391 = getitem_392 = empty_587 = None
        getitem_947: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_422['Y_ptr'];  triton_kernel_wrapper_functional_proxy_422 = None
        view_2650: "bf16[512, 1024, 196]" = torch.ops.aten.view.default(mul_392, [512, 1024, 196])
        view_2651: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_947, [512, 784, 256]);  getitem_947 = None
        view_2652: "bf16[512, 200704]" = torch.ops.aten.view.default(view_2651, [512, -1]);  view_2651 = None
        view_2653: "bf16[512, 1024, 196]" = torch.ops.aten.view.default(view_2652, [512, 1024, 196]);  view_2652 = None
        triton_kernel_wrapper_functional_proxy_423 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 525, constant_args_idx = 734, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2653, 'DY': view_2650, 'DBETA': full_default_152, 'DGAMMA': full_default_152, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_948: "f32[1024]" = triton_kernel_wrapper_functional_proxy_423['DBETA']
        getitem_949: "f32[1024]" = triton_kernel_wrapper_functional_proxy_423['DGAMMA'];  triton_kernel_wrapper_functional_proxy_423 = None
        empty_588: "bf16[512, 1024, 196]" = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_179: "bf16[512, 1024, 196]" = torch.ops.aten.permute.default(empty_588, [0, 1, 2]);  empty_588 = None
        triton_kernel_wrapper_functional_proxy_424 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 526, constant_args_idx = 735, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2653, 'DY': view_2650, 'INVSTD': rsqrt_30, 'GAMMA': primals_184, 'DBETA': getitem_948, 'DGAMMA': getitem_949, 'DX': permute_179, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2653 = view_2650 = rsqrt_30 = primals_184 = permute_179 = None
        getitem_950: "bf16[512, 1024, 196]" = triton_kernel_wrapper_functional_proxy_424['DX'];  triton_kernel_wrapper_functional_proxy_424 = None
        convert_element_type_default_61: "f32[1024]" = torch.ops.prims.convert_element_type.default(getitem_949, torch.float32);  getitem_949 = None
        convert_element_type_default_60: "f32[1024]" = torch.ops.prims.convert_element_type.default(getitem_948, torch.float32);  getitem_948 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_589: "bf16[100352, 256]" = torch.ops.aten.empty.memory_format([100352, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_425 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 527, constant_args_idx = 736, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_383, 'S_ptr': getitem_384, 'M_ptr': getitem_385, 'Y_ptr': empty_589, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_383 = getitem_384 = getitem_385 = empty_589 = None
        getitem_951: "bf16[100352, 256]" = triton_kernel_wrapper_functional_proxy_425['Y_ptr'];  triton_kernel_wrapper_functional_proxy_425 = None
        view_2670: "bf16[512, 1024, 14, 14]" = torch.ops.aten.view.default(getitem_950, [512, 1024, 14, 14]);  getitem_950 = None
        empty_590: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_45: "bf16[512, 256, 14, 14]" = torch.ops.aten.expand.default(empty_590, [512, 256, 14, 14]);  empty_590 = None
        convolution_backward_44 = torch.ops.aten.convolution_backward.default(view_2670, expand_45, convert_element_type_31, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_45 = convert_element_type_31 = None
        getitem_952: "bf16[512, 256, 14, 14]" = convolution_backward_44[0];  convolution_backward_44 = None
        empty_591: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_46: "bf16[1024, 256, 1, 1]" = torch.ops.aten.expand.default(empty_591, [1024, 256, 1, 1]);  empty_591 = None
        view_2671: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_951, [512, 196, 256]);  getitem_951 = None
        view_2672: "bf16[512, 50176]" = torch.ops.aten.view.default(view_2671, [512, -1]);  view_2671 = None
        view_2673: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(view_2672, [512, 256, 14, 14]);  view_2672 = None
        convolution_backward_45 = torch.ops.aten.convolution_backward.default(view_2670, view_2673, expand_46, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2670 = view_2673 = expand_46 = None
        getitem_956: "bf16[1024, 256, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
        convert_element_type_177: "f32[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_956, torch.float32);  getitem_956 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_426 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 737, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_382, 'Y_ptr': full_default_342, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_382 = None
        getitem_958: "i8[100352, 256]" = triton_kernel_wrapper_functional_proxy_426['Y_ptr'];  triton_kernel_wrapper_functional_proxy_426 = None
        view_2677: "i8[512, 196, 256]" = torch.ops.aten.view.default(getitem_958, [512, 196, 256]);  getitem_958 = None
        view_2678: "i8[512, 50176]" = torch.ops.aten.view.default(view_2677, [512, -1]);  view_2677 = None
        view_2679: "i8[512, 256, 14, 14]" = torch.ops.aten.view.default(view_2678, [512, 256, 14, 14]);  view_2678 = None
        mul_393: "bf16[512, 256, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_952, view_2679);  getitem_952 = view_2679 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_592: "bf16[100352, 256]" = torch.ops.aten.empty.memory_format([100352, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_427 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 528, constant_args_idx = 738, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_377, 'S_ptr': getitem_378, 'M_ptr': getitem_379, 'Y_ptr': empty_592, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_377 = getitem_378 = getitem_379 = empty_592 = None
        getitem_959: "bf16[100352, 256]" = triton_kernel_wrapper_functional_proxy_427['Y_ptr'];  triton_kernel_wrapper_functional_proxy_427 = None
        view_2695: "bf16[512, 256, 196]" = torch.ops.aten.view.default(mul_393, [512, 256, 196]);  mul_393 = None
        view_2696: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_959, [512, 196, 256]);  getitem_959 = None
        view_2697: "bf16[512, 50176]" = torch.ops.aten.view.default(view_2696, [512, -1]);  view_2696 = None
        view_2698: "bf16[512, 256, 196]" = torch.ops.aten.view.default(view_2697, [512, 256, 196]);  view_2697 = None
        triton_kernel_wrapper_functional_proxy_428 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 529, constant_args_idx = 739, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2698, 'DY': view_2695, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_960: "f32[256]" = triton_kernel_wrapper_functional_proxy_428['DBETA']
        getitem_961: "f32[256]" = triton_kernel_wrapper_functional_proxy_428['DGAMMA'];  triton_kernel_wrapper_functional_proxy_428 = None
        empty_593: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_180: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_593, [0, 1, 2]);  empty_593 = None
        triton_kernel_wrapper_functional_proxy_429 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 530, constant_args_idx = 740, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2698, 'DY': view_2695, 'INVSTD': rsqrt_29, 'GAMMA': primals_178, 'DBETA': getitem_960, 'DGAMMA': getitem_961, 'DX': permute_180, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2698 = view_2695 = rsqrt_29 = primals_178 = permute_180 = None
        getitem_962: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_429['DX'];  triton_kernel_wrapper_functional_proxy_429 = None
        convert_element_type_default_59: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_961, torch.float32);  getitem_961 = None
        convert_element_type_default_58: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_960, torch.float32);  getitem_960 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_594: "bf16[100352, 256]" = torch.ops.aten.empty.memory_format([100352, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_430 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 531, constant_args_idx = 741, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_370, 'S_ptr': getitem_371, 'M_ptr': getitem_372, 'Y_ptr': empty_594, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_370 = getitem_371 = getitem_372 = empty_594 = None
        getitem_963: "bf16[100352, 256]" = triton_kernel_wrapper_functional_proxy_430['Y_ptr'];  triton_kernel_wrapper_functional_proxy_430 = None
        view_2715: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(getitem_962, [512, 256, 14, 14]);  getitem_962 = None
        empty_595: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_47: "bf16[512, 256, 14, 14]" = torch.ops.aten.expand.default(empty_595, [512, 256, 14, 14]);  empty_595 = None
        convolution_backward_46 = torch.ops.aten.convolution_backward.default(view_2715, expand_47, convert_element_type_30, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_47 = convert_element_type_30 = None
        getitem_964: "bf16[512, 256, 14, 14]" = convolution_backward_46[0];  convolution_backward_46 = None
        empty_596: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_48: "bf16[256, 256, 3, 3]" = torch.ops.aten.expand.default(empty_596, [256, 256, 3, 3]);  empty_596 = None
        view_2716: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_963, [512, 196, 256]);  getitem_963 = None
        view_2717: "bf16[512, 50176]" = torch.ops.aten.view.default(view_2716, [512, -1]);  view_2716 = None
        view_2718: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(view_2717, [512, 256, 14, 14]);  view_2717 = None
        convolution_backward_47 = torch.ops.aten.convolution_backward.default(view_2715, view_2718, expand_48, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_2715 = view_2718 = expand_48 = None
        getitem_968: "bf16[256, 256, 3, 3]" = convolution_backward_47[1];  convolution_backward_47 = None
        convert_element_type_182: "f32[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_968, torch.float32);  getitem_968 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_431 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 742, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_369, 'Y_ptr': full_default_342, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_369 = None
        getitem_970: "i8[100352, 256]" = triton_kernel_wrapper_functional_proxy_431['Y_ptr'];  triton_kernel_wrapper_functional_proxy_431 = None
        view_2722: "i8[512, 196, 256]" = torch.ops.aten.view.default(getitem_970, [512, 196, 256]);  getitem_970 = None
        view_2723: "i8[512, 50176]" = torch.ops.aten.view.default(view_2722, [512, -1]);  view_2722 = None
        view_2724: "i8[512, 256, 14, 14]" = torch.ops.aten.view.default(view_2723, [512, 256, 14, 14]);  view_2723 = None
        mul_394: "bf16[512, 256, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_964, view_2724);  getitem_964 = view_2724 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_597: "bf16[100352, 256]" = torch.ops.aten.empty.memory_format([100352, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_432 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 532, constant_args_idx = 743, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_364, 'S_ptr': getitem_365, 'M_ptr': getitem_366, 'Y_ptr': empty_597, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_364 = getitem_365 = getitem_366 = empty_597 = None
        getitem_971: "bf16[100352, 256]" = triton_kernel_wrapper_functional_proxy_432['Y_ptr'];  triton_kernel_wrapper_functional_proxy_432 = None
        view_2740: "bf16[512, 256, 196]" = torch.ops.aten.view.default(mul_394, [512, 256, 196]);  mul_394 = None
        view_2741: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_971, [512, 196, 256]);  getitem_971 = None
        view_2742: "bf16[512, 50176]" = torch.ops.aten.view.default(view_2741, [512, -1]);  view_2741 = None
        view_2743: "bf16[512, 256, 196]" = torch.ops.aten.view.default(view_2742, [512, 256, 196]);  view_2742 = None
        triton_kernel_wrapper_functional_proxy_433 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 533, constant_args_idx = 744, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2743, 'DY': view_2740, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_972: "f32[256]" = triton_kernel_wrapper_functional_proxy_433['DBETA']
        getitem_973: "f32[256]" = triton_kernel_wrapper_functional_proxy_433['DGAMMA'];  triton_kernel_wrapper_functional_proxy_433 = None
        empty_598: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_181: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_598, [0, 1, 2]);  empty_598 = None
        triton_kernel_wrapper_functional_proxy_434 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 534, constant_args_idx = 745, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2743, 'DY': view_2740, 'INVSTD': rsqrt_28, 'GAMMA': primals_172, 'DBETA': getitem_972, 'DGAMMA': getitem_973, 'DX': permute_181, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2743 = view_2740 = rsqrt_28 = primals_172 = permute_181 = None
        getitem_974: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_434['DX'];  triton_kernel_wrapper_functional_proxy_434 = None
        convert_element_type_default_57: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_973, torch.float32);  getitem_973 = None
        convert_element_type_default_56: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_972, torch.float32);  getitem_972 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_599: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_435 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 535, constant_args_idx = 746, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_357, 'S_ptr': getitem_358, 'M_ptr': getitem_359, 'Y_ptr': empty_599, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_357 = getitem_358 = getitem_359 = empty_599 = None
        getitem_975: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_435['Y_ptr'];  triton_kernel_wrapper_functional_proxy_435 = None
        view_2760: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(getitem_974, [512, 256, 14, 14]);  getitem_974 = None
        empty_600: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_49: "bf16[512, 1024, 14, 14]" = torch.ops.aten.expand.default(empty_600, [512, 1024, 14, 14]);  empty_600 = None
        convolution_backward_48 = torch.ops.aten.convolution_backward.default(view_2760, expand_49, convert_element_type_29, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_49 = convert_element_type_29 = None
        getitem_976: "bf16[512, 1024, 14, 14]" = convolution_backward_48[0];  convolution_backward_48 = None
        empty_601: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_50: "bf16[256, 1024, 1, 1]" = torch.ops.aten.expand.default(empty_601, [256, 1024, 1, 1]);  empty_601 = None
        view_2761: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_975, [512, 784, 256]);  getitem_975 = None
        view_2762: "bf16[512, 200704]" = torch.ops.aten.view.default(view_2761, [512, -1]);  view_2761 = None
        view_2763: "bf16[512, 1024, 14, 14]" = torch.ops.aten.view.default(view_2762, [512, 1024, 14, 14]);  view_2762 = None
        convolution_backward_49 = torch.ops.aten.convolution_backward.default(view_2760, view_2763, expand_50, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2760 = view_2763 = expand_50 = None
        getitem_980: "bf16[256, 1024, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None
        convert_element_type_187: "f32[256, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_980, torch.float32);  getitem_980 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_235: "bf16[512, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_392, getitem_976);  mul_392 = getitem_976 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_436 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 747, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_356, 'Y_ptr': full_default_339, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_356 = None
        getitem_982: "i8[401408, 256]" = triton_kernel_wrapper_functional_proxy_436['Y_ptr'];  triton_kernel_wrapper_functional_proxy_436 = None
        view_2767: "i8[512, 784, 256]" = torch.ops.aten.view.default(getitem_982, [512, 784, 256]);  getitem_982 = None
        view_2768: "i8[512, 200704]" = torch.ops.aten.view.default(view_2767, [512, -1]);  view_2767 = None
        view_2769: "i8[512, 1024, 14, 14]" = torch.ops.aten.view.default(view_2768, [512, 1024, 14, 14]);  view_2768 = None
        mul_395: "bf16[512, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(add_235, view_2769);  add_235 = view_2769 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_602: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_437 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 536, constant_args_idx = 748, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_351, 'S_ptr': getitem_352, 'M_ptr': getitem_353, 'Y_ptr': empty_602, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_351 = getitem_352 = getitem_353 = empty_602 = None
        getitem_983: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_437['Y_ptr'];  triton_kernel_wrapper_functional_proxy_437 = None
        view_2785: "bf16[512, 1024, 196]" = torch.ops.aten.view.default(mul_395, [512, 1024, 196]);  mul_395 = None
        view_2786: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_983, [512, 784, 256]);  getitem_983 = None
        view_2787: "bf16[512, 200704]" = torch.ops.aten.view.default(view_2786, [512, -1]);  view_2786 = None
        view_2788: "bf16[512, 1024, 196]" = torch.ops.aten.view.default(view_2787, [512, 1024, 196]);  view_2787 = None
        triton_kernel_wrapper_functional_proxy_438 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 537, constant_args_idx = 749, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2788, 'DY': view_2785, 'DBETA': full_default_152, 'DGAMMA': full_default_152, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_984: "f32[1024]" = triton_kernel_wrapper_functional_proxy_438['DBETA']
        getitem_985: "f32[1024]" = triton_kernel_wrapper_functional_proxy_438['DGAMMA'];  triton_kernel_wrapper_functional_proxy_438 = None
        empty_603: "bf16[512, 1024, 196]" = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_182: "bf16[512, 1024, 196]" = torch.ops.aten.permute.default(empty_603, [0, 1, 2]);  empty_603 = None
        triton_kernel_wrapper_functional_proxy_439 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 538, constant_args_idx = 750, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2788, 'DY': view_2785, 'INVSTD': rsqrt_27, 'GAMMA': primals_166, 'DBETA': getitem_984, 'DGAMMA': getitem_985, 'DX': permute_182, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2788 = rsqrt_27 = primals_166 = permute_182 = None
        getitem_986: "bf16[512, 1024, 196]" = triton_kernel_wrapper_functional_proxy_439['DX'];  triton_kernel_wrapper_functional_proxy_439 = None
        convert_element_type_default_55: "f32[1024]" = torch.ops.prims.convert_element_type.default(getitem_985, torch.float32);  getitem_985 = None
        convert_element_type_default_54: "f32[1024]" = torch.ops.prims.convert_element_type.default(getitem_984, torch.float32);  getitem_984 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_604: "bf16[802816, 256]" = torch.ops.aten.empty.memory_format([802816, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_440 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 539, constant_args_idx = 751, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_344, 'S_ptr': getitem_345, 'M_ptr': getitem_346, 'Y_ptr': empty_604, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_344 = getitem_345 = getitem_346 = empty_604 = None
        getitem_987: "bf16[802816, 256]" = triton_kernel_wrapper_functional_proxy_440['Y_ptr'];  triton_kernel_wrapper_functional_proxy_440 = None
        view_2805: "bf16[512, 1024, 14, 14]" = torch.ops.aten.view.default(getitem_986, [512, 1024, 14, 14]);  getitem_986 = None
        empty_605: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_51: "bf16[512, 512, 28, 28]" = torch.ops.aten.expand.default(empty_605, [512, 512, 28, 28]);  empty_605 = None
        convolution_backward_50 = torch.ops.aten.convolution_backward.default(view_2805, expand_51, convert_element_type_28, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_51 = convert_element_type_28 = None
        getitem_988: "bf16[512, 512, 28, 28]" = convolution_backward_50[0];  convolution_backward_50 = None
        empty_606: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_52: "bf16[1024, 512, 1, 1]" = torch.ops.aten.expand.default(empty_606, [1024, 512, 1, 1]);  empty_606 = None
        view_2806: "bf16[512, 1568, 256]" = torch.ops.aten.view.default(getitem_987, [512, 1568, 256]);  getitem_987 = None
        view_2807: "bf16[512, 401408]" = torch.ops.aten.view.default(view_2806, [512, -1]);  view_2806 = None
        view_2808: "bf16[512, 512, 28, 28]" = torch.ops.aten.view.default(view_2807, [512, 512, 28, 28]);  view_2807 = None
        convolution_backward_51 = torch.ops.aten.convolution_backward.default(view_2805, view_2808, expand_52, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2805 = view_2808 = expand_52 = None
        getitem_992: "bf16[1024, 512, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
        convert_element_type_192: "f32[1024, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_992, torch.float32);  getitem_992 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_607: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_441 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 540, constant_args_idx = 752, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_341, 'S_ptr': getitem_342, 'M_ptr': getitem_343, 'Y_ptr': empty_607, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_341 = getitem_342 = getitem_343 = empty_607 = None
        getitem_994: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_441['Y_ptr'];  triton_kernel_wrapper_functional_proxy_441 = None
        view_2825: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_994, [512, 784, 256]);  getitem_994 = None
        view_2826: "bf16[512, 200704]" = torch.ops.aten.view.default(view_2825, [512, -1]);  view_2825 = None
        view_2827: "bf16[512, 1024, 196]" = torch.ops.aten.view.default(view_2826, [512, 1024, 196]);  view_2826 = None
        triton_kernel_wrapper_functional_proxy_442 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 541, constant_args_idx = 753, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2827, 'DY': view_2785, 'DBETA': full_default_152, 'DGAMMA': full_default_152, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA']);  full_default_152 = None
        getitem_995: "f32[1024]" = triton_kernel_wrapper_functional_proxy_442['DBETA']
        getitem_996: "f32[1024]" = triton_kernel_wrapper_functional_proxy_442['DGAMMA'];  triton_kernel_wrapper_functional_proxy_442 = None
        empty_608: "bf16[512, 1024, 196]" = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_183: "bf16[512, 1024, 196]" = torch.ops.aten.permute.default(empty_608, [0, 1, 2]);  empty_608 = None
        triton_kernel_wrapper_functional_proxy_443 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 542, constant_args_idx = 754, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2827, 'DY': view_2785, 'INVSTD': rsqrt_26, 'GAMMA': primals_160, 'DBETA': getitem_995, 'DGAMMA': getitem_996, 'DX': permute_183, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2827 = view_2785 = rsqrt_26 = primals_160 = permute_183 = None
        getitem_997: "bf16[512, 1024, 196]" = triton_kernel_wrapper_functional_proxy_443['DX'];  triton_kernel_wrapper_functional_proxy_443 = None
        convert_element_type_default_53: "f32[1024]" = torch.ops.prims.convert_element_type.default(getitem_996, torch.float32);  getitem_996 = None
        convert_element_type_default_52: "f32[1024]" = torch.ops.prims.convert_element_type.default(getitem_995, torch.float32);  getitem_995 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_609: "bf16[100352, 256]" = torch.ops.aten.empty.memory_format([100352, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_444 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 543, constant_args_idx = 755, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_334, 'S_ptr': getitem_335, 'M_ptr': getitem_336, 'Y_ptr': empty_609, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_334 = getitem_335 = getitem_336 = empty_609 = None
        getitem_998: "bf16[100352, 256]" = triton_kernel_wrapper_functional_proxy_444['Y_ptr'];  triton_kernel_wrapper_functional_proxy_444 = None
        view_2844: "bf16[512, 1024, 14, 14]" = torch.ops.aten.view.default(getitem_997, [512, 1024, 14, 14]);  getitem_997 = None
        empty_610: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_53: "bf16[512, 256, 14, 14]" = torch.ops.aten.expand.default(empty_610, [512, 256, 14, 14]);  empty_610 = None
        convolution_backward_52 = torch.ops.aten.convolution_backward.default(view_2844, expand_53, convert_element_type_27, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_53 = convert_element_type_27 = None
        getitem_999: "bf16[512, 256, 14, 14]" = convolution_backward_52[0];  convolution_backward_52 = None
        empty_611: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_54: "bf16[1024, 256, 1, 1]" = torch.ops.aten.expand.default(empty_611, [1024, 256, 1, 1]);  empty_611 = None
        view_2845: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_998, [512, 196, 256]);  getitem_998 = None
        view_2846: "bf16[512, 50176]" = torch.ops.aten.view.default(view_2845, [512, -1]);  view_2845 = None
        view_2847: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(view_2846, [512, 256, 14, 14]);  view_2846 = None
        convolution_backward_53 = torch.ops.aten.convolution_backward.default(view_2844, view_2847, expand_54, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2844 = view_2847 = expand_54 = None
        getitem_1003: "bf16[1024, 256, 1, 1]" = convolution_backward_53[1];  convolution_backward_53 = None
        convert_element_type_197: "f32[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1003, torch.float32);  getitem_1003 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_445 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 756, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_333, 'Y_ptr': full_default_342, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_333 = full_default_342 = None
        getitem_1005: "i8[100352, 256]" = triton_kernel_wrapper_functional_proxy_445['Y_ptr'];  triton_kernel_wrapper_functional_proxy_445 = None
        view_2851: "i8[512, 196, 256]" = torch.ops.aten.view.default(getitem_1005, [512, 196, 256]);  getitem_1005 = None
        view_2852: "i8[512, 50176]" = torch.ops.aten.view.default(view_2851, [512, -1]);  view_2851 = None
        view_2853: "i8[512, 256, 14, 14]" = torch.ops.aten.view.default(view_2852, [512, 256, 14, 14]);  view_2852 = None
        mul_396: "bf16[512, 256, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_999, view_2853);  getitem_999 = view_2853 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_612: "bf16[100352, 256]" = torch.ops.aten.empty.memory_format([100352, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_446 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 544, constant_args_idx = 757, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_328, 'S_ptr': getitem_329, 'M_ptr': getitem_330, 'Y_ptr': empty_612, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_328 = getitem_329 = getitem_330 = empty_612 = None
        getitem_1006: "bf16[100352, 256]" = triton_kernel_wrapper_functional_proxy_446['Y_ptr'];  triton_kernel_wrapper_functional_proxy_446 = None
        view_2869: "bf16[512, 256, 196]" = torch.ops.aten.view.default(mul_396, [512, 256, 196]);  mul_396 = None
        view_2870: "bf16[512, 196, 256]" = torch.ops.aten.view.default(getitem_1006, [512, 196, 256]);  getitem_1006 = None
        view_2871: "bf16[512, 50176]" = torch.ops.aten.view.default(view_2870, [512, -1]);  view_2870 = None
        view_2872: "bf16[512, 256, 196]" = torch.ops.aten.view.default(view_2871, [512, 256, 196]);  view_2871 = None
        triton_kernel_wrapper_functional_proxy_447 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 545, constant_args_idx = 758, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2872, 'DY': view_2869, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1007: "f32[256]" = triton_kernel_wrapper_functional_proxy_447['DBETA']
        getitem_1008: "f32[256]" = triton_kernel_wrapper_functional_proxy_447['DGAMMA'];  triton_kernel_wrapper_functional_proxy_447 = None
        empty_613: "bf16[512, 256, 196]" = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_184: "bf16[512, 256, 196]" = torch.ops.aten.permute.default(empty_613, [0, 1, 2]);  empty_613 = None
        triton_kernel_wrapper_functional_proxy_448 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 546, constant_args_idx = 759, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2872, 'DY': view_2869, 'INVSTD': rsqrt_25, 'GAMMA': primals_154, 'DBETA': getitem_1007, 'DGAMMA': getitem_1008, 'DX': permute_184, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2872 = view_2869 = rsqrt_25 = primals_154 = permute_184 = None
        getitem_1009: "bf16[512, 256, 196]" = triton_kernel_wrapper_functional_proxy_448['DX'];  triton_kernel_wrapper_functional_proxy_448 = None
        convert_element_type_default_51: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_1008, torch.float32);  getitem_1008 = None
        convert_element_type_default_50: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_1007, torch.float32);  getitem_1007 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_614: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_449 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 547, constant_args_idx = 760, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_321, 'S_ptr': getitem_322, 'M_ptr': getitem_323, 'Y_ptr': empty_614, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_321 = getitem_322 = getitem_323 = empty_614 = None
        getitem_1010: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_449['Y_ptr'];  triton_kernel_wrapper_functional_proxy_449 = None
        view_2889: "bf16[512, 256, 14, 14]" = torch.ops.aten.view.default(getitem_1009, [512, 256, 14, 14]);  getitem_1009 = None
        empty_615: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_55: "bf16[512, 256, 28, 28]" = torch.ops.aten.expand.default(empty_615, [512, 256, 28, 28]);  empty_615 = None
        convolution_backward_54 = torch.ops.aten.convolution_backward.default(view_2889, expand_55, convert_element_type_26, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_55 = convert_element_type_26 = None
        getitem_1011: "bf16[512, 256, 28, 28]" = convolution_backward_54[0];  convolution_backward_54 = None
        empty_616: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_56: "bf16[256, 256, 3, 3]" = torch.ops.aten.expand.default(empty_616, [256, 256, 3, 3]);  empty_616 = None
        view_2890: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_1010, [512, 784, 256]);  getitem_1010 = None
        view_2891: "bf16[512, 200704]" = torch.ops.aten.view.default(view_2890, [512, -1]);  view_2890 = None
        view_2892: "bf16[512, 256, 28, 28]" = torch.ops.aten.view.default(view_2891, [512, 256, 28, 28]);  view_2891 = None
        convolution_backward_55 = torch.ops.aten.convolution_backward.default(view_2889, view_2892, expand_56, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_2889 = view_2892 = expand_56 = None
        getitem_1015: "bf16[256, 256, 3, 3]" = convolution_backward_55[1];  convolution_backward_55 = None
        convert_element_type_202: "f32[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_1015, torch.float32);  getitem_1015 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_450 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 761, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_320, 'Y_ptr': full_default_339, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_320 = None
        getitem_1017: "i8[401408, 256]" = triton_kernel_wrapper_functional_proxy_450['Y_ptr'];  triton_kernel_wrapper_functional_proxy_450 = None
        view_2896: "i8[512, 784, 256]" = torch.ops.aten.view.default(getitem_1017, [512, 784, 256]);  getitem_1017 = None
        view_2897: "i8[512, 200704]" = torch.ops.aten.view.default(view_2896, [512, -1]);  view_2896 = None
        view_2898: "i8[512, 256, 28, 28]" = torch.ops.aten.view.default(view_2897, [512, 256, 28, 28]);  view_2897 = None
        mul_397: "bf16[512, 256, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_1011, view_2898);  getitem_1011 = view_2898 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_617: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_451 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 548, constant_args_idx = 762, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_315, 'S_ptr': getitem_316, 'M_ptr': getitem_317, 'Y_ptr': empty_617, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_315 = getitem_316 = getitem_317 = empty_617 = None
        getitem_1018: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_451['Y_ptr'];  triton_kernel_wrapper_functional_proxy_451 = None
        view_2914: "bf16[512, 256, 784]" = torch.ops.aten.view.default(mul_397, [512, 256, 784]);  mul_397 = None
        view_2915: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_1018, [512, 784, 256]);  getitem_1018 = None
        view_2916: "bf16[512, 200704]" = torch.ops.aten.view.default(view_2915, [512, -1]);  view_2915 = None
        view_2917: "bf16[512, 256, 784]" = torch.ops.aten.view.default(view_2916, [512, 256, 784]);  view_2916 = None
        triton_kernel_wrapper_functional_proxy_452 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 549, constant_args_idx = 763, grid = [(256, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2917, 'DY': view_2914, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 401408, 'HW': 784, 'stride_n': 200704, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1019: "f32[256]" = triton_kernel_wrapper_functional_proxy_452['DBETA']
        getitem_1020: "f32[256]" = triton_kernel_wrapper_functional_proxy_452['DGAMMA'];  triton_kernel_wrapper_functional_proxy_452 = None
        empty_618: "bf16[512, 256, 784]" = torch.ops.aten.empty.memory_format([512, 256, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_185: "bf16[512, 256, 784]" = torch.ops.aten.permute.default(empty_618, [0, 1, 2]);  empty_618 = None
        triton_kernel_wrapper_functional_proxy_453 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 550, constant_args_idx = 764, grid = [(256, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2917, 'DY': view_2914, 'INVSTD': rsqrt_24, 'GAMMA': primals_148, 'DBETA': getitem_1019, 'DGAMMA': getitem_1020, 'DX': permute_185, 'M': 401408, 'HW': 784, 'stride_n': 200704, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2917 = view_2914 = rsqrt_24 = primals_148 = permute_185 = None
        getitem_1021: "bf16[512, 256, 784]" = triton_kernel_wrapper_functional_proxy_453['DX'];  triton_kernel_wrapper_functional_proxy_453 = None
        convert_element_type_default_49: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_1020, torch.float32);  getitem_1020 = None
        convert_element_type_default_48: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_1019, torch.float32);  getitem_1019 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_619: "bf16[802816, 256]" = torch.ops.aten.empty.memory_format([802816, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_454 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 551, constant_args_idx = 765, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_308, 'S_ptr': getitem_309, 'M_ptr': getitem_310, 'Y_ptr': empty_619, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_308 = getitem_309 = getitem_310 = empty_619 = None
        getitem_1022: "bf16[802816, 256]" = triton_kernel_wrapper_functional_proxy_454['Y_ptr'];  triton_kernel_wrapper_functional_proxy_454 = None
        view_2934: "bf16[512, 256, 28, 28]" = torch.ops.aten.view.default(getitem_1021, [512, 256, 28, 28]);  getitem_1021 = None
        empty_620: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_57: "bf16[512, 512, 28, 28]" = torch.ops.aten.expand.default(empty_620, [512, 512, 28, 28]);  empty_620 = None
        convolution_backward_56 = torch.ops.aten.convolution_backward.default(view_2934, expand_57, convert_element_type_25, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_57 = convert_element_type_25 = None
        getitem_1023: "bf16[512, 512, 28, 28]" = convolution_backward_56[0];  convolution_backward_56 = None
        empty_621: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_58: "bf16[256, 512, 1, 1]" = torch.ops.aten.expand.default(empty_621, [256, 512, 1, 1]);  empty_621 = None
        view_2935: "bf16[512, 1568, 256]" = torch.ops.aten.view.default(getitem_1022, [512, 1568, 256]);  getitem_1022 = None
        view_2936: "bf16[512, 401408]" = torch.ops.aten.view.default(view_2935, [512, -1]);  view_2935 = None
        view_2937: "bf16[512, 512, 28, 28]" = torch.ops.aten.view.default(view_2936, [512, 512, 28, 28]);  view_2936 = None
        convolution_backward_57 = torch.ops.aten.convolution_backward.default(view_2934, view_2937, expand_58, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2934 = view_2937 = expand_58 = None
        getitem_1027: "bf16[256, 512, 1, 1]" = convolution_backward_57[1];  convolution_backward_57 = None
        convert_element_type_207: "f32[256, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1027, torch.float32);  getitem_1027 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_236: "bf16[512, 512, 28, 28]" = torch.ops.aten.add.Tensor(getitem_988, getitem_1023);  getitem_988 = getitem_1023 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_395: "i8[802816, 256]" = torch.ops.aten.full.default([802816, 256], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_455 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 766, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_307, 'Y_ptr': full_default_395, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_307 = None
        getitem_1029: "i8[802816, 256]" = triton_kernel_wrapper_functional_proxy_455['Y_ptr'];  triton_kernel_wrapper_functional_proxy_455 = None
        view_2941: "i8[512, 1568, 256]" = torch.ops.aten.view.default(getitem_1029, [512, 1568, 256]);  getitem_1029 = None
        view_2942: "i8[512, 401408]" = torch.ops.aten.view.default(view_2941, [512, -1]);  view_2941 = None
        view_2943: "i8[512, 512, 28, 28]" = torch.ops.aten.view.default(view_2942, [512, 512, 28, 28]);  view_2942 = None
        mul_398: "bf16[512, 512, 28, 28]" = torch.ops.aten.mul.Tensor(add_236, view_2943);  add_236 = view_2943 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_622: "bf16[802816, 256]" = torch.ops.aten.empty.memory_format([802816, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_456 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 552, constant_args_idx = 767, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_302, 'S_ptr': getitem_303, 'M_ptr': getitem_304, 'Y_ptr': empty_622, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_302 = getitem_303 = getitem_304 = empty_622 = None
        getitem_1030: "bf16[802816, 256]" = triton_kernel_wrapper_functional_proxy_456['Y_ptr'];  triton_kernel_wrapper_functional_proxy_456 = None
        view_2959: "bf16[512, 512, 784]" = torch.ops.aten.view.default(mul_398, [512, 512, 784])
        view_2960: "bf16[512, 1568, 256]" = torch.ops.aten.view.default(getitem_1030, [512, 1568, 256]);  getitem_1030 = None
        view_2961: "bf16[512, 401408]" = torch.ops.aten.view.default(view_2960, [512, -1]);  view_2960 = None
        view_2962: "bf16[512, 512, 784]" = torch.ops.aten.view.default(view_2961, [512, 512, 784]);  view_2961 = None
        triton_kernel_wrapper_functional_proxy_457 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 553, constant_args_idx = 768, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2962, 'DY': view_2959, 'DBETA': full_default_76, 'DGAMMA': full_default_76, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1031: "f32[512]" = triton_kernel_wrapper_functional_proxy_457['DBETA']
        getitem_1032: "f32[512]" = triton_kernel_wrapper_functional_proxy_457['DGAMMA'];  triton_kernel_wrapper_functional_proxy_457 = None
        empty_623: "bf16[512, 512, 784]" = torch.ops.aten.empty.memory_format([512, 512, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_186: "bf16[512, 512, 784]" = torch.ops.aten.permute.default(empty_623, [0, 1, 2]);  empty_623 = None
        triton_kernel_wrapper_functional_proxy_458 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 554, constant_args_idx = 769, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2962, 'DY': view_2959, 'INVSTD': rsqrt_23, 'GAMMA': primals_142, 'DBETA': getitem_1031, 'DGAMMA': getitem_1032, 'DX': permute_186, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2962 = view_2959 = rsqrt_23 = primals_142 = permute_186 = None
        getitem_1033: "bf16[512, 512, 784]" = triton_kernel_wrapper_functional_proxy_458['DX'];  triton_kernel_wrapper_functional_proxy_458 = None
        convert_element_type_default_47: "f32[512]" = torch.ops.prims.convert_element_type.default(getitem_1032, torch.float32);  getitem_1032 = None
        convert_element_type_default_46: "f32[512]" = torch.ops.prims.convert_element_type.default(getitem_1031, torch.float32);  getitem_1031 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_624: "bf16[200704, 256]" = torch.ops.aten.empty.memory_format([200704, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_459 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 555, constant_args_idx = 770, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_295, 'S_ptr': getitem_296, 'M_ptr': getitem_297, 'Y_ptr': empty_624, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_295 = getitem_296 = getitem_297 = empty_624 = None
        getitem_1034: "bf16[200704, 256]" = triton_kernel_wrapper_functional_proxy_459['Y_ptr'];  triton_kernel_wrapper_functional_proxy_459 = None
        view_2979: "bf16[512, 512, 28, 28]" = torch.ops.aten.view.default(getitem_1033, [512, 512, 28, 28]);  getitem_1033 = None
        empty_625: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_59: "bf16[512, 128, 28, 28]" = torch.ops.aten.expand.default(empty_625, [512, 128, 28, 28]);  empty_625 = None
        convolution_backward_58 = torch.ops.aten.convolution_backward.default(view_2979, expand_59, convert_element_type_24, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_59 = convert_element_type_24 = None
        getitem_1035: "bf16[512, 128, 28, 28]" = convolution_backward_58[0];  convolution_backward_58 = None
        empty_626: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_60: "bf16[512, 128, 1, 1]" = torch.ops.aten.expand.default(empty_626, [512, 128, 1, 1]);  empty_626 = None
        view_2980: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_1034, [512, 392, 256]);  getitem_1034 = None
        view_2981: "bf16[512, 100352]" = torch.ops.aten.view.default(view_2980, [512, -1]);  view_2980 = None
        view_2982: "bf16[512, 128, 28, 28]" = torch.ops.aten.view.default(view_2981, [512, 128, 28, 28]);  view_2981 = None
        convolution_backward_59 = torch.ops.aten.convolution_backward.default(view_2979, view_2982, expand_60, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2979 = view_2982 = expand_60 = None
        getitem_1039: "bf16[512, 128, 1, 1]" = convolution_backward_59[1];  convolution_backward_59 = None
        convert_element_type_212: "f32[512, 128, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1039, torch.float32);  getitem_1039 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_460 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 771, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_294, 'Y_ptr': full_default_310, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_294 = None
        getitem_1041: "i8[200704, 256]" = triton_kernel_wrapper_functional_proxy_460['Y_ptr'];  triton_kernel_wrapper_functional_proxy_460 = None
        view_2986: "i8[512, 392, 256]" = torch.ops.aten.view.default(getitem_1041, [512, 392, 256]);  getitem_1041 = None
        view_2987: "i8[512, 100352]" = torch.ops.aten.view.default(view_2986, [512, -1]);  view_2986 = None
        view_2988: "i8[512, 128, 28, 28]" = torch.ops.aten.view.default(view_2987, [512, 128, 28, 28]);  view_2987 = None
        mul_399: "bf16[512, 128, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_1035, view_2988);  getitem_1035 = view_2988 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_627: "bf16[200704, 256]" = torch.ops.aten.empty.memory_format([200704, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_461 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 556, constant_args_idx = 772, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_289, 'S_ptr': getitem_290, 'M_ptr': getitem_291, 'Y_ptr': empty_627, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_289 = getitem_290 = getitem_291 = empty_627 = None
        getitem_1042: "bf16[200704, 256]" = triton_kernel_wrapper_functional_proxy_461['Y_ptr'];  triton_kernel_wrapper_functional_proxy_461 = None
        view_3004: "bf16[512, 128, 784]" = torch.ops.aten.view.default(mul_399, [512, 128, 784]);  mul_399 = None
        view_3005: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_1042, [512, 392, 256]);  getitem_1042 = None
        view_3006: "bf16[512, 100352]" = torch.ops.aten.view.default(view_3005, [512, -1]);  view_3005 = None
        view_3007: "bf16[512, 128, 784]" = torch.ops.aten.view.default(view_3006, [512, 128, 784]);  view_3006 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        full_default_64: "f32[128]" = torch.ops.aten.full.default([128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        triton_kernel_wrapper_functional_proxy_462 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 557, constant_args_idx = 773, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3007, 'DY': view_3004, 'DBETA': full_default_64, 'DGAMMA': full_default_64, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1043: "f32[128]" = triton_kernel_wrapper_functional_proxy_462['DBETA']
        getitem_1044: "f32[128]" = triton_kernel_wrapper_functional_proxy_462['DGAMMA'];  triton_kernel_wrapper_functional_proxy_462 = None
        empty_628: "bf16[512, 128, 784]" = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_187: "bf16[512, 128, 784]" = torch.ops.aten.permute.default(empty_628, [0, 1, 2]);  empty_628 = None
        triton_kernel_wrapper_functional_proxy_463 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 558, constant_args_idx = 774, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3007, 'DY': view_3004, 'INVSTD': rsqrt_22, 'GAMMA': primals_136, 'DBETA': getitem_1043, 'DGAMMA': getitem_1044, 'DX': permute_187, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3007 = view_3004 = rsqrt_22 = primals_136 = permute_187 = None
        getitem_1045: "bf16[512, 128, 784]" = triton_kernel_wrapper_functional_proxy_463['DX'];  triton_kernel_wrapper_functional_proxy_463 = None
        convert_element_type_default_45: "f32[128]" = torch.ops.prims.convert_element_type.default(getitem_1044, torch.float32);  getitem_1044 = None
        convert_element_type_default_44: "f32[128]" = torch.ops.prims.convert_element_type.default(getitem_1043, torch.float32);  getitem_1043 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_629: "bf16[200704, 256]" = torch.ops.aten.empty.memory_format([200704, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_464 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 559, constant_args_idx = 775, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_282, 'S_ptr': getitem_283, 'M_ptr': getitem_284, 'Y_ptr': empty_629, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_282 = getitem_283 = getitem_284 = empty_629 = None
        getitem_1046: "bf16[200704, 256]" = triton_kernel_wrapper_functional_proxy_464['Y_ptr'];  triton_kernel_wrapper_functional_proxy_464 = None
        view_3024: "bf16[512, 128, 28, 28]" = torch.ops.aten.view.default(getitem_1045, [512, 128, 28, 28]);  getitem_1045 = None
        empty_630: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_61: "bf16[512, 128, 28, 28]" = torch.ops.aten.expand.default(empty_630, [512, 128, 28, 28]);  empty_630 = None
        convolution_backward_60 = torch.ops.aten.convolution_backward.default(view_3024, expand_61, convert_element_type_23, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_61 = convert_element_type_23 = None
        getitem_1047: "bf16[512, 128, 28, 28]" = convolution_backward_60[0];  convolution_backward_60 = None
        empty_631: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_62: "bf16[128, 128, 3, 3]" = torch.ops.aten.expand.default(empty_631, [128, 128, 3, 3]);  empty_631 = None
        view_3025: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_1046, [512, 392, 256]);  getitem_1046 = None
        view_3026: "bf16[512, 100352]" = torch.ops.aten.view.default(view_3025, [512, -1]);  view_3025 = None
        view_3027: "bf16[512, 128, 28, 28]" = torch.ops.aten.view.default(view_3026, [512, 128, 28, 28]);  view_3026 = None
        convolution_backward_61 = torch.ops.aten.convolution_backward.default(view_3024, view_3027, expand_62, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_3024 = view_3027 = expand_62 = None
        getitem_1051: "bf16[128, 128, 3, 3]" = convolution_backward_61[1];  convolution_backward_61 = None
        convert_element_type_217: "f32[128, 128, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_1051, torch.float32);  getitem_1051 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_465 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 776, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_281, 'Y_ptr': full_default_310, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_281 = None
        getitem_1053: "i8[200704, 256]" = triton_kernel_wrapper_functional_proxy_465['Y_ptr'];  triton_kernel_wrapper_functional_proxy_465 = None
        view_3031: "i8[512, 392, 256]" = torch.ops.aten.view.default(getitem_1053, [512, 392, 256]);  getitem_1053 = None
        view_3032: "i8[512, 100352]" = torch.ops.aten.view.default(view_3031, [512, -1]);  view_3031 = None
        view_3033: "i8[512, 128, 28, 28]" = torch.ops.aten.view.default(view_3032, [512, 128, 28, 28]);  view_3032 = None
        mul_400: "bf16[512, 128, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_1047, view_3033);  getitem_1047 = view_3033 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_632: "bf16[200704, 256]" = torch.ops.aten.empty.memory_format([200704, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_466 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 560, constant_args_idx = 777, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_276, 'S_ptr': getitem_277, 'M_ptr': getitem_278, 'Y_ptr': empty_632, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_276 = getitem_277 = getitem_278 = empty_632 = None
        getitem_1054: "bf16[200704, 256]" = triton_kernel_wrapper_functional_proxy_466['Y_ptr'];  triton_kernel_wrapper_functional_proxy_466 = None
        view_3049: "bf16[512, 128, 784]" = torch.ops.aten.view.default(mul_400, [512, 128, 784]);  mul_400 = None
        view_3050: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_1054, [512, 392, 256]);  getitem_1054 = None
        view_3051: "bf16[512, 100352]" = torch.ops.aten.view.default(view_3050, [512, -1]);  view_3050 = None
        view_3052: "bf16[512, 128, 784]" = torch.ops.aten.view.default(view_3051, [512, 128, 784]);  view_3051 = None
        triton_kernel_wrapper_functional_proxy_467 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 561, constant_args_idx = 778, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3052, 'DY': view_3049, 'DBETA': full_default_64, 'DGAMMA': full_default_64, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1055: "f32[128]" = triton_kernel_wrapper_functional_proxy_467['DBETA']
        getitem_1056: "f32[128]" = triton_kernel_wrapper_functional_proxy_467['DGAMMA'];  triton_kernel_wrapper_functional_proxy_467 = None
        empty_633: "bf16[512, 128, 784]" = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_188: "bf16[512, 128, 784]" = torch.ops.aten.permute.default(empty_633, [0, 1, 2]);  empty_633 = None
        triton_kernel_wrapper_functional_proxy_468 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 562, constant_args_idx = 779, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3052, 'DY': view_3049, 'INVSTD': rsqrt_21, 'GAMMA': primals_130, 'DBETA': getitem_1055, 'DGAMMA': getitem_1056, 'DX': permute_188, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3052 = view_3049 = rsqrt_21 = primals_130 = permute_188 = None
        getitem_1057: "bf16[512, 128, 784]" = triton_kernel_wrapper_functional_proxy_468['DX'];  triton_kernel_wrapper_functional_proxy_468 = None
        convert_element_type_default_43: "f32[128]" = torch.ops.prims.convert_element_type.default(getitem_1056, torch.float32);  getitem_1056 = None
        convert_element_type_default_42: "f32[128]" = torch.ops.prims.convert_element_type.default(getitem_1055, torch.float32);  getitem_1055 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_634: "bf16[802816, 256]" = torch.ops.aten.empty.memory_format([802816, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_469 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 563, constant_args_idx = 780, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_269, 'S_ptr': getitem_270, 'M_ptr': getitem_271, 'Y_ptr': empty_634, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_269 = getitem_270 = getitem_271 = empty_634 = None
        getitem_1058: "bf16[802816, 256]" = triton_kernel_wrapper_functional_proxy_469['Y_ptr'];  triton_kernel_wrapper_functional_proxy_469 = None
        view_3069: "bf16[512, 128, 28, 28]" = torch.ops.aten.view.default(getitem_1057, [512, 128, 28, 28]);  getitem_1057 = None
        empty_635: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_63: "bf16[512, 512, 28, 28]" = torch.ops.aten.expand.default(empty_635, [512, 512, 28, 28]);  empty_635 = None
        convolution_backward_62 = torch.ops.aten.convolution_backward.default(view_3069, expand_63, convert_element_type_22, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_63 = convert_element_type_22 = None
        getitem_1059: "bf16[512, 512, 28, 28]" = convolution_backward_62[0];  convolution_backward_62 = None
        empty_636: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_64: "bf16[128, 512, 1, 1]" = torch.ops.aten.expand.default(empty_636, [128, 512, 1, 1]);  empty_636 = None
        view_3070: "bf16[512, 1568, 256]" = torch.ops.aten.view.default(getitem_1058, [512, 1568, 256]);  getitem_1058 = None
        view_3071: "bf16[512, 401408]" = torch.ops.aten.view.default(view_3070, [512, -1]);  view_3070 = None
        view_3072: "bf16[512, 512, 28, 28]" = torch.ops.aten.view.default(view_3071, [512, 512, 28, 28]);  view_3071 = None
        convolution_backward_63 = torch.ops.aten.convolution_backward.default(view_3069, view_3072, expand_64, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_3069 = view_3072 = expand_64 = None
        getitem_1063: "bf16[128, 512, 1, 1]" = convolution_backward_63[1];  convolution_backward_63 = None
        convert_element_type_222: "f32[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1063, torch.float32);  getitem_1063 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_237: "bf16[512, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_398, getitem_1059);  mul_398 = getitem_1059 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_470 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 781, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_268, 'Y_ptr': full_default_395, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_268 = None
        getitem_1065: "i8[802816, 256]" = triton_kernel_wrapper_functional_proxy_470['Y_ptr'];  triton_kernel_wrapper_functional_proxy_470 = None
        view_3076: "i8[512, 1568, 256]" = torch.ops.aten.view.default(getitem_1065, [512, 1568, 256]);  getitem_1065 = None
        view_3077: "i8[512, 401408]" = torch.ops.aten.view.default(view_3076, [512, -1]);  view_3076 = None
        view_3078: "i8[512, 512, 28, 28]" = torch.ops.aten.view.default(view_3077, [512, 512, 28, 28]);  view_3077 = None
        mul_401: "bf16[512, 512, 28, 28]" = torch.ops.aten.mul.Tensor(add_237, view_3078);  add_237 = view_3078 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_637: "bf16[802816, 256]" = torch.ops.aten.empty.memory_format([802816, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_471 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 564, constant_args_idx = 782, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_263, 'S_ptr': getitem_264, 'M_ptr': getitem_265, 'Y_ptr': empty_637, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_263 = getitem_264 = getitem_265 = empty_637 = None
        getitem_1066: "bf16[802816, 256]" = triton_kernel_wrapper_functional_proxy_471['Y_ptr'];  triton_kernel_wrapper_functional_proxy_471 = None
        view_3094: "bf16[512, 512, 784]" = torch.ops.aten.view.default(mul_401, [512, 512, 784])
        view_3095: "bf16[512, 1568, 256]" = torch.ops.aten.view.default(getitem_1066, [512, 1568, 256]);  getitem_1066 = None
        view_3096: "bf16[512, 401408]" = torch.ops.aten.view.default(view_3095, [512, -1]);  view_3095 = None
        view_3097: "bf16[512, 512, 784]" = torch.ops.aten.view.default(view_3096, [512, 512, 784]);  view_3096 = None
        triton_kernel_wrapper_functional_proxy_472 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 565, constant_args_idx = 783, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3097, 'DY': view_3094, 'DBETA': full_default_76, 'DGAMMA': full_default_76, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1067: "f32[512]" = triton_kernel_wrapper_functional_proxy_472['DBETA']
        getitem_1068: "f32[512]" = triton_kernel_wrapper_functional_proxy_472['DGAMMA'];  triton_kernel_wrapper_functional_proxy_472 = None
        empty_638: "bf16[512, 512, 784]" = torch.ops.aten.empty.memory_format([512, 512, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_189: "bf16[512, 512, 784]" = torch.ops.aten.permute.default(empty_638, [0, 1, 2]);  empty_638 = None
        triton_kernel_wrapper_functional_proxy_473 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 566, constant_args_idx = 784, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3097, 'DY': view_3094, 'INVSTD': rsqrt_20, 'GAMMA': primals_124, 'DBETA': getitem_1067, 'DGAMMA': getitem_1068, 'DX': permute_189, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3097 = view_3094 = rsqrt_20 = primals_124 = permute_189 = None
        getitem_1069: "bf16[512, 512, 784]" = triton_kernel_wrapper_functional_proxy_473['DX'];  triton_kernel_wrapper_functional_proxy_473 = None
        convert_element_type_default_41: "f32[512]" = torch.ops.prims.convert_element_type.default(getitem_1068, torch.float32);  getitem_1068 = None
        convert_element_type_default_40: "f32[512]" = torch.ops.prims.convert_element_type.default(getitem_1067, torch.float32);  getitem_1067 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_639: "bf16[200704, 256]" = torch.ops.aten.empty.memory_format([200704, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_474 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 567, constant_args_idx = 785, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_256, 'S_ptr': getitem_257, 'M_ptr': getitem_258, 'Y_ptr': empty_639, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_256 = getitem_257 = getitem_258 = empty_639 = None
        getitem_1070: "bf16[200704, 256]" = triton_kernel_wrapper_functional_proxy_474['Y_ptr'];  triton_kernel_wrapper_functional_proxy_474 = None
        view_3114: "bf16[512, 512, 28, 28]" = torch.ops.aten.view.default(getitem_1069, [512, 512, 28, 28]);  getitem_1069 = None
        empty_640: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_65: "bf16[512, 128, 28, 28]" = torch.ops.aten.expand.default(empty_640, [512, 128, 28, 28]);  empty_640 = None
        convolution_backward_64 = torch.ops.aten.convolution_backward.default(view_3114, expand_65, convert_element_type_21, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_65 = convert_element_type_21 = None
        getitem_1071: "bf16[512, 128, 28, 28]" = convolution_backward_64[0];  convolution_backward_64 = None
        empty_641: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_66: "bf16[512, 128, 1, 1]" = torch.ops.aten.expand.default(empty_641, [512, 128, 1, 1]);  empty_641 = None
        view_3115: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_1070, [512, 392, 256]);  getitem_1070 = None
        view_3116: "bf16[512, 100352]" = torch.ops.aten.view.default(view_3115, [512, -1]);  view_3115 = None
        view_3117: "bf16[512, 128, 28, 28]" = torch.ops.aten.view.default(view_3116, [512, 128, 28, 28]);  view_3116 = None
        convolution_backward_65 = torch.ops.aten.convolution_backward.default(view_3114, view_3117, expand_66, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_3114 = view_3117 = expand_66 = None
        getitem_1075: "bf16[512, 128, 1, 1]" = convolution_backward_65[1];  convolution_backward_65 = None
        convert_element_type_227: "f32[512, 128, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1075, torch.float32);  getitem_1075 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_475 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 786, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_255, 'Y_ptr': full_default_310, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_255 = None
        getitem_1077: "i8[200704, 256]" = triton_kernel_wrapper_functional_proxy_475['Y_ptr'];  triton_kernel_wrapper_functional_proxy_475 = None
        view_3121: "i8[512, 392, 256]" = torch.ops.aten.view.default(getitem_1077, [512, 392, 256]);  getitem_1077 = None
        view_3122: "i8[512, 100352]" = torch.ops.aten.view.default(view_3121, [512, -1]);  view_3121 = None
        view_3123: "i8[512, 128, 28, 28]" = torch.ops.aten.view.default(view_3122, [512, 128, 28, 28]);  view_3122 = None
        mul_402: "bf16[512, 128, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_1071, view_3123);  getitem_1071 = view_3123 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_642: "bf16[200704, 256]" = torch.ops.aten.empty.memory_format([200704, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_476 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 568, constant_args_idx = 787, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_250, 'S_ptr': getitem_251, 'M_ptr': getitem_252, 'Y_ptr': empty_642, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_250 = getitem_251 = getitem_252 = empty_642 = None
        getitem_1078: "bf16[200704, 256]" = triton_kernel_wrapper_functional_proxy_476['Y_ptr'];  triton_kernel_wrapper_functional_proxy_476 = None
        view_3139: "bf16[512, 128, 784]" = torch.ops.aten.view.default(mul_402, [512, 128, 784]);  mul_402 = None
        view_3140: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_1078, [512, 392, 256]);  getitem_1078 = None
        view_3141: "bf16[512, 100352]" = torch.ops.aten.view.default(view_3140, [512, -1]);  view_3140 = None
        view_3142: "bf16[512, 128, 784]" = torch.ops.aten.view.default(view_3141, [512, 128, 784]);  view_3141 = None
        triton_kernel_wrapper_functional_proxy_477 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 569, constant_args_idx = 788, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3142, 'DY': view_3139, 'DBETA': full_default_64, 'DGAMMA': full_default_64, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1079: "f32[128]" = triton_kernel_wrapper_functional_proxy_477['DBETA']
        getitem_1080: "f32[128]" = triton_kernel_wrapper_functional_proxy_477['DGAMMA'];  triton_kernel_wrapper_functional_proxy_477 = None
        empty_643: "bf16[512, 128, 784]" = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_190: "bf16[512, 128, 784]" = torch.ops.aten.permute.default(empty_643, [0, 1, 2]);  empty_643 = None
        triton_kernel_wrapper_functional_proxy_478 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 570, constant_args_idx = 789, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3142, 'DY': view_3139, 'INVSTD': rsqrt_19, 'GAMMA': primals_118, 'DBETA': getitem_1079, 'DGAMMA': getitem_1080, 'DX': permute_190, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3142 = view_3139 = rsqrt_19 = primals_118 = permute_190 = None
        getitem_1081: "bf16[512, 128, 784]" = triton_kernel_wrapper_functional_proxy_478['DX'];  triton_kernel_wrapper_functional_proxy_478 = None
        convert_element_type_default_39: "f32[128]" = torch.ops.prims.convert_element_type.default(getitem_1080, torch.float32);  getitem_1080 = None
        convert_element_type_default_38: "f32[128]" = torch.ops.prims.convert_element_type.default(getitem_1079, torch.float32);  getitem_1079 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_644: "bf16[200704, 256]" = torch.ops.aten.empty.memory_format([200704, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_479 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 571, constant_args_idx = 790, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_243, 'S_ptr': getitem_244, 'M_ptr': getitem_245, 'Y_ptr': empty_644, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_243 = getitem_244 = getitem_245 = empty_644 = None
        getitem_1082: "bf16[200704, 256]" = triton_kernel_wrapper_functional_proxy_479['Y_ptr'];  triton_kernel_wrapper_functional_proxy_479 = None
        view_3159: "bf16[512, 128, 28, 28]" = torch.ops.aten.view.default(getitem_1081, [512, 128, 28, 28]);  getitem_1081 = None
        empty_645: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_67: "bf16[512, 128, 28, 28]" = torch.ops.aten.expand.default(empty_645, [512, 128, 28, 28]);  empty_645 = None
        convolution_backward_66 = torch.ops.aten.convolution_backward.default(view_3159, expand_67, convert_element_type_20, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_67 = convert_element_type_20 = None
        getitem_1083: "bf16[512, 128, 28, 28]" = convolution_backward_66[0];  convolution_backward_66 = None
        empty_646: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_68: "bf16[128, 128, 3, 3]" = torch.ops.aten.expand.default(empty_646, [128, 128, 3, 3]);  empty_646 = None
        view_3160: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_1082, [512, 392, 256]);  getitem_1082 = None
        view_3161: "bf16[512, 100352]" = torch.ops.aten.view.default(view_3160, [512, -1]);  view_3160 = None
        view_3162: "bf16[512, 128, 28, 28]" = torch.ops.aten.view.default(view_3161, [512, 128, 28, 28]);  view_3161 = None
        convolution_backward_67 = torch.ops.aten.convolution_backward.default(view_3159, view_3162, expand_68, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_3159 = view_3162 = expand_68 = None
        getitem_1087: "bf16[128, 128, 3, 3]" = convolution_backward_67[1];  convolution_backward_67 = None
        convert_element_type_232: "f32[128, 128, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_1087, torch.float32);  getitem_1087 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_480 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 791, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_242, 'Y_ptr': full_default_310, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_242 = None
        getitem_1089: "i8[200704, 256]" = triton_kernel_wrapper_functional_proxy_480['Y_ptr'];  triton_kernel_wrapper_functional_proxy_480 = None
        view_3166: "i8[512, 392, 256]" = torch.ops.aten.view.default(getitem_1089, [512, 392, 256]);  getitem_1089 = None
        view_3167: "i8[512, 100352]" = torch.ops.aten.view.default(view_3166, [512, -1]);  view_3166 = None
        view_3168: "i8[512, 128, 28, 28]" = torch.ops.aten.view.default(view_3167, [512, 128, 28, 28]);  view_3167 = None
        mul_403: "bf16[512, 128, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_1083, view_3168);  getitem_1083 = view_3168 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_647: "bf16[200704, 256]" = torch.ops.aten.empty.memory_format([200704, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_481 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 572, constant_args_idx = 792, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_237, 'S_ptr': getitem_238, 'M_ptr': getitem_239, 'Y_ptr': empty_647, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_237 = getitem_238 = getitem_239 = empty_647 = None
        getitem_1090: "bf16[200704, 256]" = triton_kernel_wrapper_functional_proxy_481['Y_ptr'];  triton_kernel_wrapper_functional_proxy_481 = None
        view_3184: "bf16[512, 128, 784]" = torch.ops.aten.view.default(mul_403, [512, 128, 784]);  mul_403 = None
        view_3185: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_1090, [512, 392, 256]);  getitem_1090 = None
        view_3186: "bf16[512, 100352]" = torch.ops.aten.view.default(view_3185, [512, -1]);  view_3185 = None
        view_3187: "bf16[512, 128, 784]" = torch.ops.aten.view.default(view_3186, [512, 128, 784]);  view_3186 = None
        triton_kernel_wrapper_functional_proxy_482 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 573, constant_args_idx = 793, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3187, 'DY': view_3184, 'DBETA': full_default_64, 'DGAMMA': full_default_64, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1091: "f32[128]" = triton_kernel_wrapper_functional_proxy_482['DBETA']
        getitem_1092: "f32[128]" = triton_kernel_wrapper_functional_proxy_482['DGAMMA'];  triton_kernel_wrapper_functional_proxy_482 = None
        empty_648: "bf16[512, 128, 784]" = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_191: "bf16[512, 128, 784]" = torch.ops.aten.permute.default(empty_648, [0, 1, 2]);  empty_648 = None
        triton_kernel_wrapper_functional_proxy_483 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 574, constant_args_idx = 794, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3187, 'DY': view_3184, 'INVSTD': rsqrt_18, 'GAMMA': primals_112, 'DBETA': getitem_1091, 'DGAMMA': getitem_1092, 'DX': permute_191, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3187 = view_3184 = rsqrt_18 = primals_112 = permute_191 = None
        getitem_1093: "bf16[512, 128, 784]" = triton_kernel_wrapper_functional_proxy_483['DX'];  triton_kernel_wrapper_functional_proxy_483 = None
        convert_element_type_default_37: "f32[128]" = torch.ops.prims.convert_element_type.default(getitem_1092, torch.float32);  getitem_1092 = None
        convert_element_type_default_36: "f32[128]" = torch.ops.prims.convert_element_type.default(getitem_1091, torch.float32);  getitem_1091 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_649: "bf16[802816, 256]" = torch.ops.aten.empty.memory_format([802816, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_484 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 575, constant_args_idx = 795, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_230, 'S_ptr': getitem_231, 'M_ptr': getitem_232, 'Y_ptr': empty_649, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_230 = getitem_231 = getitem_232 = empty_649 = None
        getitem_1094: "bf16[802816, 256]" = triton_kernel_wrapper_functional_proxy_484['Y_ptr'];  triton_kernel_wrapper_functional_proxy_484 = None
        view_3204: "bf16[512, 128, 28, 28]" = torch.ops.aten.view.default(getitem_1093, [512, 128, 28, 28]);  getitem_1093 = None
        empty_650: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_69: "bf16[512, 512, 28, 28]" = torch.ops.aten.expand.default(empty_650, [512, 512, 28, 28]);  empty_650 = None
        convolution_backward_68 = torch.ops.aten.convolution_backward.default(view_3204, expand_69, convert_element_type_19, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_69 = convert_element_type_19 = None
        getitem_1095: "bf16[512, 512, 28, 28]" = convolution_backward_68[0];  convolution_backward_68 = None
        empty_651: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_70: "bf16[128, 512, 1, 1]" = torch.ops.aten.expand.default(empty_651, [128, 512, 1, 1]);  empty_651 = None
        view_3205: "bf16[512, 1568, 256]" = torch.ops.aten.view.default(getitem_1094, [512, 1568, 256]);  getitem_1094 = None
        view_3206: "bf16[512, 401408]" = torch.ops.aten.view.default(view_3205, [512, -1]);  view_3205 = None
        view_3207: "bf16[512, 512, 28, 28]" = torch.ops.aten.view.default(view_3206, [512, 512, 28, 28]);  view_3206 = None
        convolution_backward_69 = torch.ops.aten.convolution_backward.default(view_3204, view_3207, expand_70, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_3204 = view_3207 = expand_70 = None
        getitem_1099: "bf16[128, 512, 1, 1]" = convolution_backward_69[1];  convolution_backward_69 = None
        convert_element_type_237: "f32[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1099, torch.float32);  getitem_1099 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_238: "bf16[512, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_401, getitem_1095);  mul_401 = getitem_1095 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_485 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 796, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_229, 'Y_ptr': full_default_395, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_229 = None
        getitem_1101: "i8[802816, 256]" = triton_kernel_wrapper_functional_proxy_485['Y_ptr'];  triton_kernel_wrapper_functional_proxy_485 = None
        view_3211: "i8[512, 1568, 256]" = torch.ops.aten.view.default(getitem_1101, [512, 1568, 256]);  getitem_1101 = None
        view_3212: "i8[512, 401408]" = torch.ops.aten.view.default(view_3211, [512, -1]);  view_3211 = None
        view_3213: "i8[512, 512, 28, 28]" = torch.ops.aten.view.default(view_3212, [512, 512, 28, 28]);  view_3212 = None
        mul_404: "bf16[512, 512, 28, 28]" = torch.ops.aten.mul.Tensor(add_238, view_3213);  add_238 = view_3213 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_652: "bf16[802816, 256]" = torch.ops.aten.empty.memory_format([802816, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_486 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 576, constant_args_idx = 797, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_224, 'S_ptr': getitem_225, 'M_ptr': getitem_226, 'Y_ptr': empty_652, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_224 = getitem_225 = getitem_226 = empty_652 = None
        getitem_1102: "bf16[802816, 256]" = triton_kernel_wrapper_functional_proxy_486['Y_ptr'];  triton_kernel_wrapper_functional_proxy_486 = None
        view_3229: "bf16[512, 512, 784]" = torch.ops.aten.view.default(mul_404, [512, 512, 784])
        view_3230: "bf16[512, 1568, 256]" = torch.ops.aten.view.default(getitem_1102, [512, 1568, 256]);  getitem_1102 = None
        view_3231: "bf16[512, 401408]" = torch.ops.aten.view.default(view_3230, [512, -1]);  view_3230 = None
        view_3232: "bf16[512, 512, 784]" = torch.ops.aten.view.default(view_3231, [512, 512, 784]);  view_3231 = None
        triton_kernel_wrapper_functional_proxy_487 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 577, constant_args_idx = 798, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3232, 'DY': view_3229, 'DBETA': full_default_76, 'DGAMMA': full_default_76, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1103: "f32[512]" = triton_kernel_wrapper_functional_proxy_487['DBETA']
        getitem_1104: "f32[512]" = triton_kernel_wrapper_functional_proxy_487['DGAMMA'];  triton_kernel_wrapper_functional_proxy_487 = None
        empty_653: "bf16[512, 512, 784]" = torch.ops.aten.empty.memory_format([512, 512, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_192: "bf16[512, 512, 784]" = torch.ops.aten.permute.default(empty_653, [0, 1, 2]);  empty_653 = None
        triton_kernel_wrapper_functional_proxy_488 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 578, constant_args_idx = 799, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3232, 'DY': view_3229, 'INVSTD': rsqrt_17, 'GAMMA': primals_106, 'DBETA': getitem_1103, 'DGAMMA': getitem_1104, 'DX': permute_192, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3232 = view_3229 = rsqrt_17 = primals_106 = permute_192 = None
        getitem_1105: "bf16[512, 512, 784]" = triton_kernel_wrapper_functional_proxy_488['DX'];  triton_kernel_wrapper_functional_proxy_488 = None
        convert_element_type_default_35: "f32[512]" = torch.ops.prims.convert_element_type.default(getitem_1104, torch.float32);  getitem_1104 = None
        convert_element_type_default_34: "f32[512]" = torch.ops.prims.convert_element_type.default(getitem_1103, torch.float32);  getitem_1103 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_654: "bf16[200704, 256]" = torch.ops.aten.empty.memory_format([200704, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_489 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 579, constant_args_idx = 800, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_217, 'S_ptr': getitem_218, 'M_ptr': getitem_219, 'Y_ptr': empty_654, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_217 = getitem_218 = getitem_219 = empty_654 = None
        getitem_1106: "bf16[200704, 256]" = triton_kernel_wrapper_functional_proxy_489['Y_ptr'];  triton_kernel_wrapper_functional_proxy_489 = None
        view_3249: "bf16[512, 512, 28, 28]" = torch.ops.aten.view.default(getitem_1105, [512, 512, 28, 28]);  getitem_1105 = None
        empty_655: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_71: "bf16[512, 128, 28, 28]" = torch.ops.aten.expand.default(empty_655, [512, 128, 28, 28]);  empty_655 = None
        convolution_backward_70 = torch.ops.aten.convolution_backward.default(view_3249, expand_71, convert_element_type_18, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_71 = convert_element_type_18 = None
        getitem_1107: "bf16[512, 128, 28, 28]" = convolution_backward_70[0];  convolution_backward_70 = None
        empty_656: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_72: "bf16[512, 128, 1, 1]" = torch.ops.aten.expand.default(empty_656, [512, 128, 1, 1]);  empty_656 = None
        view_3250: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_1106, [512, 392, 256]);  getitem_1106 = None
        view_3251: "bf16[512, 100352]" = torch.ops.aten.view.default(view_3250, [512, -1]);  view_3250 = None
        view_3252: "bf16[512, 128, 28, 28]" = torch.ops.aten.view.default(view_3251, [512, 128, 28, 28]);  view_3251 = None
        convolution_backward_71 = torch.ops.aten.convolution_backward.default(view_3249, view_3252, expand_72, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_3249 = view_3252 = expand_72 = None
        getitem_1111: "bf16[512, 128, 1, 1]" = convolution_backward_71[1];  convolution_backward_71 = None
        convert_element_type_242: "f32[512, 128, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1111, torch.float32);  getitem_1111 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_490 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 801, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_216, 'Y_ptr': full_default_310, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_216 = None
        getitem_1113: "i8[200704, 256]" = triton_kernel_wrapper_functional_proxy_490['Y_ptr'];  triton_kernel_wrapper_functional_proxy_490 = None
        view_3256: "i8[512, 392, 256]" = torch.ops.aten.view.default(getitem_1113, [512, 392, 256]);  getitem_1113 = None
        view_3257: "i8[512, 100352]" = torch.ops.aten.view.default(view_3256, [512, -1]);  view_3256 = None
        view_3258: "i8[512, 128, 28, 28]" = torch.ops.aten.view.default(view_3257, [512, 128, 28, 28]);  view_3257 = None
        mul_405: "bf16[512, 128, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_1107, view_3258);  getitem_1107 = view_3258 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_657: "bf16[200704, 256]" = torch.ops.aten.empty.memory_format([200704, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_491 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 580, constant_args_idx = 802, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_211, 'S_ptr': getitem_212, 'M_ptr': getitem_213, 'Y_ptr': empty_657, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_211 = getitem_212 = getitem_213 = empty_657 = None
        getitem_1114: "bf16[200704, 256]" = triton_kernel_wrapper_functional_proxy_491['Y_ptr'];  triton_kernel_wrapper_functional_proxy_491 = None
        view_3274: "bf16[512, 128, 784]" = torch.ops.aten.view.default(mul_405, [512, 128, 784]);  mul_405 = None
        view_3275: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_1114, [512, 392, 256]);  getitem_1114 = None
        view_3276: "bf16[512, 100352]" = torch.ops.aten.view.default(view_3275, [512, -1]);  view_3275 = None
        view_3277: "bf16[512, 128, 784]" = torch.ops.aten.view.default(view_3276, [512, 128, 784]);  view_3276 = None
        triton_kernel_wrapper_functional_proxy_492 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 581, constant_args_idx = 803, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3277, 'DY': view_3274, 'DBETA': full_default_64, 'DGAMMA': full_default_64, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1115: "f32[128]" = triton_kernel_wrapper_functional_proxy_492['DBETA']
        getitem_1116: "f32[128]" = triton_kernel_wrapper_functional_proxy_492['DGAMMA'];  triton_kernel_wrapper_functional_proxy_492 = None
        empty_658: "bf16[512, 128, 784]" = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_193: "bf16[512, 128, 784]" = torch.ops.aten.permute.default(empty_658, [0, 1, 2]);  empty_658 = None
        triton_kernel_wrapper_functional_proxy_493 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 582, constant_args_idx = 804, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3277, 'DY': view_3274, 'INVSTD': rsqrt_16, 'GAMMA': primals_100, 'DBETA': getitem_1115, 'DGAMMA': getitem_1116, 'DX': permute_193, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3277 = view_3274 = rsqrt_16 = primals_100 = permute_193 = None
        getitem_1117: "bf16[512, 128, 784]" = triton_kernel_wrapper_functional_proxy_493['DX'];  triton_kernel_wrapper_functional_proxy_493 = None
        convert_element_type_default_33: "f32[128]" = torch.ops.prims.convert_element_type.default(getitem_1116, torch.float32);  getitem_1116 = None
        convert_element_type_default_32: "f32[128]" = torch.ops.prims.convert_element_type.default(getitem_1115, torch.float32);  getitem_1115 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_659: "bf16[200704, 256]" = torch.ops.aten.empty.memory_format([200704, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_494 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 583, constant_args_idx = 805, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_204, 'S_ptr': getitem_205, 'M_ptr': getitem_206, 'Y_ptr': empty_659, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_204 = getitem_205 = getitem_206 = empty_659 = None
        getitem_1118: "bf16[200704, 256]" = triton_kernel_wrapper_functional_proxy_494['Y_ptr'];  triton_kernel_wrapper_functional_proxy_494 = None
        view_3294: "bf16[512, 128, 28, 28]" = torch.ops.aten.view.default(getitem_1117, [512, 128, 28, 28]);  getitem_1117 = None
        empty_660: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_73: "bf16[512, 128, 28, 28]" = torch.ops.aten.expand.default(empty_660, [512, 128, 28, 28]);  empty_660 = None
        convolution_backward_72 = torch.ops.aten.convolution_backward.default(view_3294, expand_73, convert_element_type_17, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_73 = convert_element_type_17 = None
        getitem_1119: "bf16[512, 128, 28, 28]" = convolution_backward_72[0];  convolution_backward_72 = None
        empty_661: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_74: "bf16[128, 128, 3, 3]" = torch.ops.aten.expand.default(empty_661, [128, 128, 3, 3]);  empty_661 = None
        view_3295: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_1118, [512, 392, 256]);  getitem_1118 = None
        view_3296: "bf16[512, 100352]" = torch.ops.aten.view.default(view_3295, [512, -1]);  view_3295 = None
        view_3297: "bf16[512, 128, 28, 28]" = torch.ops.aten.view.default(view_3296, [512, 128, 28, 28]);  view_3296 = None
        convolution_backward_73 = torch.ops.aten.convolution_backward.default(view_3294, view_3297, expand_74, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_3294 = view_3297 = expand_74 = None
        getitem_1123: "bf16[128, 128, 3, 3]" = convolution_backward_73[1];  convolution_backward_73 = None
        convert_element_type_247: "f32[128, 128, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_1123, torch.float32);  getitem_1123 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_495 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 806, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_203, 'Y_ptr': full_default_310, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_203 = None
        getitem_1125: "i8[200704, 256]" = triton_kernel_wrapper_functional_proxy_495['Y_ptr'];  triton_kernel_wrapper_functional_proxy_495 = None
        view_3301: "i8[512, 392, 256]" = torch.ops.aten.view.default(getitem_1125, [512, 392, 256]);  getitem_1125 = None
        view_3302: "i8[512, 100352]" = torch.ops.aten.view.default(view_3301, [512, -1]);  view_3301 = None
        view_3303: "i8[512, 128, 28, 28]" = torch.ops.aten.view.default(view_3302, [512, 128, 28, 28]);  view_3302 = None
        mul_406: "bf16[512, 128, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_1119, view_3303);  getitem_1119 = view_3303 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_662: "bf16[200704, 256]" = torch.ops.aten.empty.memory_format([200704, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_496 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 584, constant_args_idx = 807, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_198, 'S_ptr': getitem_199, 'M_ptr': getitem_200, 'Y_ptr': empty_662, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_198 = getitem_199 = getitem_200 = empty_662 = None
        getitem_1126: "bf16[200704, 256]" = triton_kernel_wrapper_functional_proxy_496['Y_ptr'];  triton_kernel_wrapper_functional_proxy_496 = None
        view_3319: "bf16[512, 128, 784]" = torch.ops.aten.view.default(mul_406, [512, 128, 784]);  mul_406 = None
        view_3320: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_1126, [512, 392, 256]);  getitem_1126 = None
        view_3321: "bf16[512, 100352]" = torch.ops.aten.view.default(view_3320, [512, -1]);  view_3320 = None
        view_3322: "bf16[512, 128, 784]" = torch.ops.aten.view.default(view_3321, [512, 128, 784]);  view_3321 = None
        triton_kernel_wrapper_functional_proxy_497 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 585, constant_args_idx = 808, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3322, 'DY': view_3319, 'DBETA': full_default_64, 'DGAMMA': full_default_64, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1127: "f32[128]" = triton_kernel_wrapper_functional_proxy_497['DBETA']
        getitem_1128: "f32[128]" = triton_kernel_wrapper_functional_proxy_497['DGAMMA'];  triton_kernel_wrapper_functional_proxy_497 = None
        empty_663: "bf16[512, 128, 784]" = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_194: "bf16[512, 128, 784]" = torch.ops.aten.permute.default(empty_663, [0, 1, 2]);  empty_663 = None
        triton_kernel_wrapper_functional_proxy_498 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 586, constant_args_idx = 809, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3322, 'DY': view_3319, 'INVSTD': rsqrt_15, 'GAMMA': primals_94, 'DBETA': getitem_1127, 'DGAMMA': getitem_1128, 'DX': permute_194, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3322 = view_3319 = rsqrt_15 = primals_94 = permute_194 = None
        getitem_1129: "bf16[512, 128, 784]" = triton_kernel_wrapper_functional_proxy_498['DX'];  triton_kernel_wrapper_functional_proxy_498 = None
        convert_element_type_default_31: "f32[128]" = torch.ops.prims.convert_element_type.default(getitem_1128, torch.float32);  getitem_1128 = None
        convert_element_type_default_30: "f32[128]" = torch.ops.prims.convert_element_type.default(getitem_1127, torch.float32);  getitem_1127 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_664: "bf16[802816, 256]" = torch.ops.aten.empty.memory_format([802816, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_499 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 587, constant_args_idx = 810, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_191, 'S_ptr': getitem_192, 'M_ptr': getitem_193, 'Y_ptr': empty_664, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_191 = getitem_192 = getitem_193 = empty_664 = None
        getitem_1130: "bf16[802816, 256]" = triton_kernel_wrapper_functional_proxy_499['Y_ptr'];  triton_kernel_wrapper_functional_proxy_499 = None
        view_3339: "bf16[512, 128, 28, 28]" = torch.ops.aten.view.default(getitem_1129, [512, 128, 28, 28]);  getitem_1129 = None
        empty_665: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_75: "bf16[512, 512, 28, 28]" = torch.ops.aten.expand.default(empty_665, [512, 512, 28, 28]);  empty_665 = None
        convolution_backward_74 = torch.ops.aten.convolution_backward.default(view_3339, expand_75, convert_element_type_16, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_75 = convert_element_type_16 = None
        getitem_1131: "bf16[512, 512, 28, 28]" = convolution_backward_74[0];  convolution_backward_74 = None
        empty_666: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_76: "bf16[128, 512, 1, 1]" = torch.ops.aten.expand.default(empty_666, [128, 512, 1, 1]);  empty_666 = None
        view_3340: "bf16[512, 1568, 256]" = torch.ops.aten.view.default(getitem_1130, [512, 1568, 256]);  getitem_1130 = None
        view_3341: "bf16[512, 401408]" = torch.ops.aten.view.default(view_3340, [512, -1]);  view_3340 = None
        view_3342: "bf16[512, 512, 28, 28]" = torch.ops.aten.view.default(view_3341, [512, 512, 28, 28]);  view_3341 = None
        convolution_backward_75 = torch.ops.aten.convolution_backward.default(view_3339, view_3342, expand_76, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_3339 = view_3342 = expand_76 = None
        getitem_1135: "bf16[128, 512, 1, 1]" = convolution_backward_75[1];  convolution_backward_75 = None
        convert_element_type_252: "f32[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1135, torch.float32);  getitem_1135 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_239: "bf16[512, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_404, getitem_1131);  mul_404 = getitem_1131 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_500 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 811, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_190, 'Y_ptr': full_default_395, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_190 = None
        getitem_1137: "i8[802816, 256]" = triton_kernel_wrapper_functional_proxy_500['Y_ptr'];  triton_kernel_wrapper_functional_proxy_500 = None
        view_3346: "i8[512, 1568, 256]" = torch.ops.aten.view.default(getitem_1137, [512, 1568, 256]);  getitem_1137 = None
        view_3347: "i8[512, 401408]" = torch.ops.aten.view.default(view_3346, [512, -1]);  view_3346 = None
        view_3348: "i8[512, 512, 28, 28]" = torch.ops.aten.view.default(view_3347, [512, 512, 28, 28]);  view_3347 = None
        mul_407: "bf16[512, 512, 28, 28]" = torch.ops.aten.mul.Tensor(add_239, view_3348);  add_239 = view_3348 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_667: "bf16[802816, 256]" = torch.ops.aten.empty.memory_format([802816, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_501 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 588, constant_args_idx = 812, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_185, 'S_ptr': getitem_186, 'M_ptr': getitem_187, 'Y_ptr': empty_667, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_185 = getitem_186 = getitem_187 = empty_667 = None
        getitem_1138: "bf16[802816, 256]" = triton_kernel_wrapper_functional_proxy_501['Y_ptr'];  triton_kernel_wrapper_functional_proxy_501 = None
        view_3364: "bf16[512, 512, 784]" = torch.ops.aten.view.default(mul_407, [512, 512, 784]);  mul_407 = None
        view_3365: "bf16[512, 1568, 256]" = torch.ops.aten.view.default(getitem_1138, [512, 1568, 256]);  getitem_1138 = None
        view_3366: "bf16[512, 401408]" = torch.ops.aten.view.default(view_3365, [512, -1]);  view_3365 = None
        view_3367: "bf16[512, 512, 784]" = torch.ops.aten.view.default(view_3366, [512, 512, 784]);  view_3366 = None
        triton_kernel_wrapper_functional_proxy_502 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 589, constant_args_idx = 813, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3367, 'DY': view_3364, 'DBETA': full_default_76, 'DGAMMA': full_default_76, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1139: "f32[512]" = triton_kernel_wrapper_functional_proxy_502['DBETA']
        getitem_1140: "f32[512]" = triton_kernel_wrapper_functional_proxy_502['DGAMMA'];  triton_kernel_wrapper_functional_proxy_502 = None
        empty_668: "bf16[512, 512, 784]" = torch.ops.aten.empty.memory_format([512, 512, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_195: "bf16[512, 512, 784]" = torch.ops.aten.permute.default(empty_668, [0, 1, 2]);  empty_668 = None
        triton_kernel_wrapper_functional_proxy_503 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 590, constant_args_idx = 814, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3367, 'DY': view_3364, 'INVSTD': rsqrt_14, 'GAMMA': primals_88, 'DBETA': getitem_1139, 'DGAMMA': getitem_1140, 'DX': permute_195, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3367 = rsqrt_14 = primals_88 = permute_195 = None
        getitem_1141: "bf16[512, 512, 784]" = triton_kernel_wrapper_functional_proxy_503['DX'];  triton_kernel_wrapper_functional_proxy_503 = None
        convert_element_type_default_29: "f32[512]" = torch.ops.prims.convert_element_type.default(getitem_1140, torch.float32);  getitem_1140 = None
        convert_element_type_default_28: "f32[512]" = torch.ops.prims.convert_element_type.default(getitem_1139, torch.float32);  getitem_1139 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_669: "bf16[1605632, 256]" = torch.ops.aten.empty.memory_format([1605632, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_504 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 591, constant_args_idx = 815, grid = [(1605632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_178, 'S_ptr': getitem_179, 'M_ptr': getitem_180, 'Y_ptr': empty_669, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_178 = getitem_179 = getitem_180 = empty_669 = None
        getitem_1142: "bf16[1605632, 256]" = triton_kernel_wrapper_functional_proxy_504['Y_ptr'];  triton_kernel_wrapper_functional_proxy_504 = None
        view_3384: "bf16[512, 512, 28, 28]" = torch.ops.aten.view.default(getitem_1141, [512, 512, 28, 28]);  getitem_1141 = None
        empty_670: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_77: "bf16[512, 256, 56, 56]" = torch.ops.aten.expand.default(empty_670, [512, 256, 56, 56]);  empty_670 = None
        convolution_backward_76 = torch.ops.aten.convolution_backward.default(view_3384, expand_77, convert_element_type_15, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_77 = convert_element_type_15 = None
        getitem_1143: "bf16[512, 256, 56, 56]" = convolution_backward_76[0];  convolution_backward_76 = None
        empty_671: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_78: "bf16[512, 256, 1, 1]" = torch.ops.aten.expand.default(empty_671, [512, 256, 1, 1]);  empty_671 = None
        view_3385: "bf16[512, 3136, 256]" = torch.ops.aten.view.default(getitem_1142, [512, 3136, 256]);  getitem_1142 = None
        view_3386: "bf16[512, 802816]" = torch.ops.aten.view.default(view_3385, [512, -1]);  view_3385 = None
        view_3387: "bf16[512, 256, 56, 56]" = torch.ops.aten.view.default(view_3386, [512, 256, 56, 56]);  view_3386 = None
        convolution_backward_77 = torch.ops.aten.convolution_backward.default(view_3384, view_3387, expand_78, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_3384 = view_3387 = expand_78 = None
        getitem_1147: "bf16[512, 256, 1, 1]" = convolution_backward_77[1];  convolution_backward_77 = None
        convert_element_type_257: "f32[512, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1147, torch.float32);  getitem_1147 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_672: "bf16[802816, 256]" = torch.ops.aten.empty.memory_format([802816, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_505 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 592, constant_args_idx = 816, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_175, 'S_ptr': getitem_176, 'M_ptr': getitem_177, 'Y_ptr': empty_672, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_175 = getitem_176 = getitem_177 = empty_672 = None
        getitem_1149: "bf16[802816, 256]" = triton_kernel_wrapper_functional_proxy_505['Y_ptr'];  triton_kernel_wrapper_functional_proxy_505 = None
        view_3404: "bf16[512, 1568, 256]" = torch.ops.aten.view.default(getitem_1149, [512, 1568, 256]);  getitem_1149 = None
        view_3405: "bf16[512, 401408]" = torch.ops.aten.view.default(view_3404, [512, -1]);  view_3404 = None
        view_3406: "bf16[512, 512, 784]" = torch.ops.aten.view.default(view_3405, [512, 512, 784]);  view_3405 = None
        triton_kernel_wrapper_functional_proxy_506 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 593, constant_args_idx = 817, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3406, 'DY': view_3364, 'DBETA': full_default_76, 'DGAMMA': full_default_76, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA']);  full_default_76 = None
        getitem_1150: "f32[512]" = triton_kernel_wrapper_functional_proxy_506['DBETA']
        getitem_1151: "f32[512]" = triton_kernel_wrapper_functional_proxy_506['DGAMMA'];  triton_kernel_wrapper_functional_proxy_506 = None
        empty_673: "bf16[512, 512, 784]" = torch.ops.aten.empty.memory_format([512, 512, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_196: "bf16[512, 512, 784]" = torch.ops.aten.permute.default(empty_673, [0, 1, 2]);  empty_673 = None
        triton_kernel_wrapper_functional_proxy_507 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 594, constant_args_idx = 818, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3406, 'DY': view_3364, 'INVSTD': rsqrt_13, 'GAMMA': primals_82, 'DBETA': getitem_1150, 'DGAMMA': getitem_1151, 'DX': permute_196, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3406 = view_3364 = rsqrt_13 = primals_82 = permute_196 = None
        getitem_1152: "bf16[512, 512, 784]" = triton_kernel_wrapper_functional_proxy_507['DX'];  triton_kernel_wrapper_functional_proxy_507 = None
        convert_element_type_default_27: "f32[512]" = torch.ops.prims.convert_element_type.default(getitem_1151, torch.float32);  getitem_1151 = None
        convert_element_type_default_26: "f32[512]" = torch.ops.prims.convert_element_type.default(getitem_1150, torch.float32);  getitem_1150 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_674: "bf16[200704, 256]" = torch.ops.aten.empty.memory_format([200704, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_508 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 595, constant_args_idx = 819, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_168, 'S_ptr': getitem_169, 'M_ptr': getitem_170, 'Y_ptr': empty_674, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_168 = getitem_169 = getitem_170 = empty_674 = None
        getitem_1153: "bf16[200704, 256]" = triton_kernel_wrapper_functional_proxy_508['Y_ptr'];  triton_kernel_wrapper_functional_proxy_508 = None
        view_3423: "bf16[512, 512, 28, 28]" = torch.ops.aten.view.default(getitem_1152, [512, 512, 28, 28]);  getitem_1152 = None
        empty_675: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_79: "bf16[512, 128, 28, 28]" = torch.ops.aten.expand.default(empty_675, [512, 128, 28, 28]);  empty_675 = None
        convolution_backward_78 = torch.ops.aten.convolution_backward.default(view_3423, expand_79, convert_element_type_14, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_79 = convert_element_type_14 = None
        getitem_1154: "bf16[512, 128, 28, 28]" = convolution_backward_78[0];  convolution_backward_78 = None
        empty_676: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_80: "bf16[512, 128, 1, 1]" = torch.ops.aten.expand.default(empty_676, [512, 128, 1, 1]);  empty_676 = None
        view_3424: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_1153, [512, 392, 256]);  getitem_1153 = None
        view_3425: "bf16[512, 100352]" = torch.ops.aten.view.default(view_3424, [512, -1]);  view_3424 = None
        view_3426: "bf16[512, 128, 28, 28]" = torch.ops.aten.view.default(view_3425, [512, 128, 28, 28]);  view_3425 = None
        convolution_backward_79 = torch.ops.aten.convolution_backward.default(view_3423, view_3426, expand_80, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_3423 = view_3426 = expand_80 = None
        getitem_1158: "bf16[512, 128, 1, 1]" = convolution_backward_79[1];  convolution_backward_79 = None
        convert_element_type_262: "f32[512, 128, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1158, torch.float32);  getitem_1158 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_509 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 820, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_167, 'Y_ptr': full_default_310, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_167 = full_default_310 = None
        getitem_1160: "i8[200704, 256]" = triton_kernel_wrapper_functional_proxy_509['Y_ptr'];  triton_kernel_wrapper_functional_proxy_509 = None
        view_3430: "i8[512, 392, 256]" = torch.ops.aten.view.default(getitem_1160, [512, 392, 256]);  getitem_1160 = None
        view_3431: "i8[512, 100352]" = torch.ops.aten.view.default(view_3430, [512, -1]);  view_3430 = None
        view_3432: "i8[512, 128, 28, 28]" = torch.ops.aten.view.default(view_3431, [512, 128, 28, 28]);  view_3431 = None
        mul_408: "bf16[512, 128, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_1154, view_3432);  getitem_1154 = view_3432 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_677: "bf16[200704, 256]" = torch.ops.aten.empty.memory_format([200704, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_510 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 596, constant_args_idx = 821, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_162, 'S_ptr': getitem_163, 'M_ptr': getitem_164, 'Y_ptr': empty_677, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_162 = getitem_163 = getitem_164 = empty_677 = None
        getitem_1161: "bf16[200704, 256]" = triton_kernel_wrapper_functional_proxy_510['Y_ptr'];  triton_kernel_wrapper_functional_proxy_510 = None
        view_3448: "bf16[512, 128, 784]" = torch.ops.aten.view.default(mul_408, [512, 128, 784]);  mul_408 = None
        view_3449: "bf16[512, 392, 256]" = torch.ops.aten.view.default(getitem_1161, [512, 392, 256]);  getitem_1161 = None
        view_3450: "bf16[512, 100352]" = torch.ops.aten.view.default(view_3449, [512, -1]);  view_3449 = None
        view_3451: "bf16[512, 128, 784]" = torch.ops.aten.view.default(view_3450, [512, 128, 784]);  view_3450 = None
        triton_kernel_wrapper_functional_proxy_511 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 597, constant_args_idx = 822, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3451, 'DY': view_3448, 'DBETA': full_default_64, 'DGAMMA': full_default_64, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1162: "f32[128]" = triton_kernel_wrapper_functional_proxy_511['DBETA']
        getitem_1163: "f32[128]" = triton_kernel_wrapper_functional_proxy_511['DGAMMA'];  triton_kernel_wrapper_functional_proxy_511 = None
        empty_678: "bf16[512, 128, 784]" = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_197: "bf16[512, 128, 784]" = torch.ops.aten.permute.default(empty_678, [0, 1, 2]);  empty_678 = None
        triton_kernel_wrapper_functional_proxy_512 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 598, constant_args_idx = 823, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3451, 'DY': view_3448, 'INVSTD': rsqrt_12, 'GAMMA': primals_76, 'DBETA': getitem_1162, 'DGAMMA': getitem_1163, 'DX': permute_197, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3451 = view_3448 = rsqrt_12 = primals_76 = permute_197 = None
        getitem_1164: "bf16[512, 128, 784]" = triton_kernel_wrapper_functional_proxy_512['DX'];  triton_kernel_wrapper_functional_proxy_512 = None
        convert_element_type_default_25: "f32[128]" = torch.ops.prims.convert_element_type.default(getitem_1163, torch.float32);  getitem_1163 = None
        convert_element_type_default_24: "f32[128]" = torch.ops.prims.convert_element_type.default(getitem_1162, torch.float32);  getitem_1162 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_679: "bf16[802816, 256]" = torch.ops.aten.empty.memory_format([802816, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_513 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 599, constant_args_idx = 824, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_155, 'S_ptr': getitem_156, 'M_ptr': getitem_157, 'Y_ptr': empty_679, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_155 = getitem_156 = getitem_157 = empty_679 = None
        getitem_1165: "bf16[802816, 256]" = triton_kernel_wrapper_functional_proxy_513['Y_ptr'];  triton_kernel_wrapper_functional_proxy_513 = None
        view_3468: "bf16[512, 128, 28, 28]" = torch.ops.aten.view.default(getitem_1164, [512, 128, 28, 28]);  getitem_1164 = None
        empty_680: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_81: "bf16[512, 128, 56, 56]" = torch.ops.aten.expand.default(empty_680, [512, 128, 56, 56]);  empty_680 = None
        convolution_backward_80 = torch.ops.aten.convolution_backward.default(view_3468, expand_81, convert_element_type_13, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_81 = convert_element_type_13 = None
        getitem_1166: "bf16[512, 128, 56, 56]" = convolution_backward_80[0];  convolution_backward_80 = None
        empty_681: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_82: "bf16[128, 128, 3, 3]" = torch.ops.aten.expand.default(empty_681, [128, 128, 3, 3]);  empty_681 = None
        view_3469: "bf16[512, 1568, 256]" = torch.ops.aten.view.default(getitem_1165, [512, 1568, 256]);  getitem_1165 = None
        view_3470: "bf16[512, 401408]" = torch.ops.aten.view.default(view_3469, [512, -1]);  view_3469 = None
        view_3471: "bf16[512, 128, 56, 56]" = torch.ops.aten.view.default(view_3470, [512, 128, 56, 56]);  view_3470 = None
        convolution_backward_81 = torch.ops.aten.convolution_backward.default(view_3468, view_3471, expand_82, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_3468 = view_3471 = expand_82 = None
        getitem_1170: "bf16[128, 128, 3, 3]" = convolution_backward_81[1];  convolution_backward_81 = None
        convert_element_type_267: "f32[128, 128, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_1170, torch.float32);  getitem_1170 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_514 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 825, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_154, 'Y_ptr': full_default_395, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_154 = full_default_395 = None
        getitem_1172: "i8[802816, 256]" = triton_kernel_wrapper_functional_proxy_514['Y_ptr'];  triton_kernel_wrapper_functional_proxy_514 = None
        view_3475: "i8[512, 1568, 256]" = torch.ops.aten.view.default(getitem_1172, [512, 1568, 256]);  getitem_1172 = None
        view_3476: "i8[512, 401408]" = torch.ops.aten.view.default(view_3475, [512, -1]);  view_3475 = None
        view_3477: "i8[512, 128, 56, 56]" = torch.ops.aten.view.default(view_3476, [512, 128, 56, 56]);  view_3476 = None
        mul_409: "bf16[512, 128, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_1166, view_3477);  getitem_1166 = view_3477 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_682: "bf16[802816, 256]" = torch.ops.aten.empty.memory_format([802816, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_515 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 600, constant_args_idx = 826, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_149, 'S_ptr': getitem_150, 'M_ptr': getitem_151, 'Y_ptr': empty_682, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_149 = getitem_150 = getitem_151 = empty_682 = None
        getitem_1173: "bf16[802816, 256]" = triton_kernel_wrapper_functional_proxy_515['Y_ptr'];  triton_kernel_wrapper_functional_proxy_515 = None
        view_3493: "bf16[512, 128, 3136]" = torch.ops.aten.view.default(mul_409, [512, 128, 3136]);  mul_409 = None
        view_3494: "bf16[512, 1568, 256]" = torch.ops.aten.view.default(getitem_1173, [512, 1568, 256]);  getitem_1173 = None
        view_3495: "bf16[512, 401408]" = torch.ops.aten.view.default(view_3494, [512, -1]);  view_3494 = None
        view_3496: "bf16[512, 128, 3136]" = torch.ops.aten.view.default(view_3495, [512, 128, 3136]);  view_3495 = None
        triton_kernel_wrapper_functional_proxy_516 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 601, constant_args_idx = 827, grid = [(128, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3496, 'DY': view_3493, 'DBETA': full_default_64, 'DGAMMA': full_default_64, 'M': 1605632, 'HW': 3136, 'stride_n': 401408, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA']);  full_default_64 = None
        getitem_1174: "f32[128]" = triton_kernel_wrapper_functional_proxy_516['DBETA']
        getitem_1175: "f32[128]" = triton_kernel_wrapper_functional_proxy_516['DGAMMA'];  triton_kernel_wrapper_functional_proxy_516 = None
        empty_683: "bf16[512, 128, 3136]" = torch.ops.aten.empty.memory_format([512, 128, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_198: "bf16[512, 128, 3136]" = torch.ops.aten.permute.default(empty_683, [0, 1, 2]);  empty_683 = None
        triton_kernel_wrapper_functional_proxy_517 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 602, constant_args_idx = 828, grid = [(128, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3496, 'DY': view_3493, 'INVSTD': rsqrt_11, 'GAMMA': primals_70, 'DBETA': getitem_1174, 'DGAMMA': getitem_1175, 'DX': permute_198, 'M': 1605632, 'HW': 3136, 'stride_n': 401408, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3496 = view_3493 = rsqrt_11 = primals_70 = permute_198 = None
        getitem_1176: "bf16[512, 128, 3136]" = triton_kernel_wrapper_functional_proxy_517['DX'];  triton_kernel_wrapper_functional_proxy_517 = None
        convert_element_type_default_23: "f32[128]" = torch.ops.prims.convert_element_type.default(getitem_1175, torch.float32);  getitem_1175 = None
        convert_element_type_default_22: "f32[128]" = torch.ops.prims.convert_element_type.default(getitem_1174, torch.float32);  getitem_1174 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_684: "bf16[1605632, 256]" = torch.ops.aten.empty.memory_format([1605632, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_518 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 603, constant_args_idx = 829, grid = [(1605632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_142, 'S_ptr': getitem_143, 'M_ptr': getitem_144, 'Y_ptr': empty_684, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_142 = getitem_143 = getitem_144 = empty_684 = None
        getitem_1177: "bf16[1605632, 256]" = triton_kernel_wrapper_functional_proxy_518['Y_ptr'];  triton_kernel_wrapper_functional_proxy_518 = None
        view_3513: "bf16[512, 128, 56, 56]" = torch.ops.aten.view.default(getitem_1176, [512, 128, 56, 56]);  getitem_1176 = None
        empty_685: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_83: "bf16[512, 256, 56, 56]" = torch.ops.aten.expand.default(empty_685, [512, 256, 56, 56]);  empty_685 = None
        convolution_backward_82 = torch.ops.aten.convolution_backward.default(view_3513, expand_83, convert_element_type_12, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_83 = convert_element_type_12 = None
        getitem_1178: "bf16[512, 256, 56, 56]" = convolution_backward_82[0];  convolution_backward_82 = None
        empty_686: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_84: "bf16[128, 256, 1, 1]" = torch.ops.aten.expand.default(empty_686, [128, 256, 1, 1]);  empty_686 = None
        view_3514: "bf16[512, 3136, 256]" = torch.ops.aten.view.default(getitem_1177, [512, 3136, 256]);  getitem_1177 = None
        view_3515: "bf16[512, 802816]" = torch.ops.aten.view.default(view_3514, [512, -1]);  view_3514 = None
        view_3516: "bf16[512, 256, 56, 56]" = torch.ops.aten.view.default(view_3515, [512, 256, 56, 56]);  view_3515 = None
        convolution_backward_83 = torch.ops.aten.convolution_backward.default(view_3513, view_3516, expand_84, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_3513 = view_3516 = expand_84 = None
        getitem_1182: "bf16[128, 256, 1, 1]" = convolution_backward_83[1];  convolution_backward_83 = None
        convert_element_type_272: "f32[128, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1182, torch.float32);  getitem_1182 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_240: "bf16[512, 256, 56, 56]" = torch.ops.aten.add.Tensor(getitem_1143, getitem_1178);  getitem_1143 = getitem_1178 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        full_default_433: "i8[1605632, 256]" = torch.ops.aten.full.default([1605632, 256], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_519 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 830, grid = [(1605632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_141, 'Y_ptr': full_default_433, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_141 = None
        getitem_1184: "i8[1605632, 256]" = triton_kernel_wrapper_functional_proxy_519['Y_ptr'];  triton_kernel_wrapper_functional_proxy_519 = None
        view_3520: "i8[512, 3136, 256]" = torch.ops.aten.view.default(getitem_1184, [512, 3136, 256]);  getitem_1184 = None
        view_3521: "i8[512, 802816]" = torch.ops.aten.view.default(view_3520, [512, -1]);  view_3520 = None
        view_3522: "i8[512, 256, 56, 56]" = torch.ops.aten.view.default(view_3521, [512, 256, 56, 56]);  view_3521 = None
        mul_410: "bf16[512, 256, 56, 56]" = torch.ops.aten.mul.Tensor(add_240, view_3522);  add_240 = view_3522 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_687: "bf16[1605632, 256]" = torch.ops.aten.empty.memory_format([1605632, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_520 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 604, constant_args_idx = 831, grid = [(1605632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_136, 'S_ptr': getitem_137, 'M_ptr': getitem_138, 'Y_ptr': empty_687, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_136 = getitem_137 = getitem_138 = empty_687 = None
        getitem_1185: "bf16[1605632, 256]" = triton_kernel_wrapper_functional_proxy_520['Y_ptr'];  triton_kernel_wrapper_functional_proxy_520 = None
        view_3538: "bf16[512, 256, 3136]" = torch.ops.aten.view.default(mul_410, [512, 256, 3136])
        view_3539: "bf16[512, 3136, 256]" = torch.ops.aten.view.default(getitem_1185, [512, 3136, 256]);  getitem_1185 = None
        view_3540: "bf16[512, 802816]" = torch.ops.aten.view.default(view_3539, [512, -1]);  view_3539 = None
        view_3541: "bf16[512, 256, 3136]" = torch.ops.aten.view.default(view_3540, [512, 256, 3136]);  view_3540 = None
        triton_kernel_wrapper_functional_proxy_521 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 605, constant_args_idx = 832, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3541, 'DY': view_3538, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1186: "f32[256]" = triton_kernel_wrapper_functional_proxy_521['DBETA']
        getitem_1187: "f32[256]" = triton_kernel_wrapper_functional_proxy_521['DGAMMA'];  triton_kernel_wrapper_functional_proxy_521 = None
        empty_688: "bf16[512, 256, 3136]" = torch.ops.aten.empty.memory_format([512, 256, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_199: "bf16[512, 256, 3136]" = torch.ops.aten.permute.default(empty_688, [0, 1, 2]);  empty_688 = None
        triton_kernel_wrapper_functional_proxy_522 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 606, constant_args_idx = 833, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3541, 'DY': view_3538, 'INVSTD': rsqrt_10, 'GAMMA': primals_64, 'DBETA': getitem_1186, 'DGAMMA': getitem_1187, 'DX': permute_199, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3541 = view_3538 = rsqrt_10 = primals_64 = permute_199 = None
        getitem_1188: "bf16[512, 256, 3136]" = triton_kernel_wrapper_functional_proxy_522['DX'];  triton_kernel_wrapper_functional_proxy_522 = None
        convert_element_type_default_21: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_1187, torch.float32);  getitem_1187 = None
        convert_element_type_default_20: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_1186, torch.float32);  getitem_1186 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_689: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_523 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 607, constant_args_idx = 834, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_129, 'S_ptr': getitem_130, 'M_ptr': getitem_131, 'Y_ptr': empty_689, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_129 = getitem_130 = getitem_131 = empty_689 = None
        getitem_1189: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_523['Y_ptr'];  triton_kernel_wrapper_functional_proxy_523 = None
        view_3558: "bf16[512, 256, 56, 56]" = torch.ops.aten.view.default(getitem_1188, [512, 256, 56, 56]);  getitem_1188 = None
        empty_690: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_85: "bf16[512, 64, 56, 56]" = torch.ops.aten.expand.default(empty_690, [512, 64, 56, 56]);  empty_690 = None
        convolution_backward_84 = torch.ops.aten.convolution_backward.default(view_3558, expand_85, convert_element_type_11, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_85 = convert_element_type_11 = None
        getitem_1190: "bf16[512, 64, 56, 56]" = convolution_backward_84[0];  convolution_backward_84 = None
        empty_691: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_86: "bf16[256, 64, 1, 1]" = torch.ops.aten.expand.default(empty_691, [256, 64, 1, 1]);  empty_691 = None
        view_3559: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_1189, [512, 784, 256]);  getitem_1189 = None
        view_3560: "bf16[512, 200704]" = torch.ops.aten.view.default(view_3559, [512, -1]);  view_3559 = None
        view_3561: "bf16[512, 64, 56, 56]" = torch.ops.aten.view.default(view_3560, [512, 64, 56, 56]);  view_3560 = None
        convolution_backward_85 = torch.ops.aten.convolution_backward.default(view_3558, view_3561, expand_86, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_3558 = view_3561 = expand_86 = None
        getitem_1194: "bf16[256, 64, 1, 1]" = convolution_backward_85[1];  convolution_backward_85 = None
        convert_element_type_277: "f32[256, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1194, torch.float32);  getitem_1194 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_524 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 835, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_128, 'Y_ptr': full_default_339, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_128 = None
        getitem_1196: "i8[401408, 256]" = triton_kernel_wrapper_functional_proxy_524['Y_ptr'];  triton_kernel_wrapper_functional_proxy_524 = None
        view_3565: "i8[512, 784, 256]" = torch.ops.aten.view.default(getitem_1196, [512, 784, 256]);  getitem_1196 = None
        view_3566: "i8[512, 200704]" = torch.ops.aten.view.default(view_3565, [512, -1]);  view_3565 = None
        view_3567: "i8[512, 64, 56, 56]" = torch.ops.aten.view.default(view_3566, [512, 64, 56, 56]);  view_3566 = None
        mul_411: "bf16[512, 64, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_1190, view_3567);  getitem_1190 = view_3567 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_692: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_525 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 608, constant_args_idx = 836, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_123, 'S_ptr': getitem_124, 'M_ptr': getitem_125, 'Y_ptr': empty_692, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_123 = getitem_124 = getitem_125 = empty_692 = None
        getitem_1197: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_525['Y_ptr'];  triton_kernel_wrapper_functional_proxy_525 = None
        view_3583: "bf16[512, 64, 3136]" = torch.ops.aten.view.default(mul_411, [512, 64, 3136]);  mul_411 = None
        view_3584: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_1197, [512, 784, 256]);  getitem_1197 = None
        view_3585: "bf16[512, 200704]" = torch.ops.aten.view.default(view_3584, [512, -1]);  view_3584 = None
        view_3586: "bf16[512, 64, 3136]" = torch.ops.aten.view.default(view_3585, [512, 64, 3136]);  view_3585 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        full_default: "f32[64]" = torch.ops.aten.full.default([64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        triton_kernel_wrapper_functional_proxy_526 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 609, constant_args_idx = 837, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3586, 'DY': view_3583, 'DBETA': full_default, 'DGAMMA': full_default, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1198: "f32[64]" = triton_kernel_wrapper_functional_proxy_526['DBETA']
        getitem_1199: "f32[64]" = triton_kernel_wrapper_functional_proxy_526['DGAMMA'];  triton_kernel_wrapper_functional_proxy_526 = None
        empty_693: "bf16[512, 64, 3136]" = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_200: "bf16[512, 64, 3136]" = torch.ops.aten.permute.default(empty_693, [0, 1, 2]);  empty_693 = None
        triton_kernel_wrapper_functional_proxy_527 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 610, constant_args_idx = 838, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3586, 'DY': view_3583, 'INVSTD': rsqrt_9, 'GAMMA': primals_58, 'DBETA': getitem_1198, 'DGAMMA': getitem_1199, 'DX': permute_200, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3586 = view_3583 = rsqrt_9 = primals_58 = permute_200 = None
        getitem_1200: "bf16[512, 64, 3136]" = triton_kernel_wrapper_functional_proxy_527['DX'];  triton_kernel_wrapper_functional_proxy_527 = None
        convert_element_type_default_19: "f32[64]" = torch.ops.prims.convert_element_type.default(getitem_1199, torch.float32);  getitem_1199 = None
        convert_element_type_default_18: "f32[64]" = torch.ops.prims.convert_element_type.default(getitem_1198, torch.float32);  getitem_1198 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_694: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_528 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 611, constant_args_idx = 839, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_116, 'S_ptr': getitem_117, 'M_ptr': getitem_118, 'Y_ptr': empty_694, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_116 = getitem_117 = getitem_118 = empty_694 = None
        getitem_1201: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_528['Y_ptr'];  triton_kernel_wrapper_functional_proxy_528 = None
        view_3603: "bf16[512, 64, 56, 56]" = torch.ops.aten.view.default(getitem_1200, [512, 64, 56, 56]);  getitem_1200 = None
        empty_695: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_87: "bf16[512, 64, 56, 56]" = torch.ops.aten.expand.default(empty_695, [512, 64, 56, 56]);  empty_695 = None
        convolution_backward_86 = torch.ops.aten.convolution_backward.default(view_3603, expand_87, convert_element_type_10, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_87 = convert_element_type_10 = None
        getitem_1202: "bf16[512, 64, 56, 56]" = convolution_backward_86[0];  convolution_backward_86 = None
        empty_696: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_88: "bf16[64, 64, 3, 3]" = torch.ops.aten.expand.default(empty_696, [64, 64, 3, 3]);  empty_696 = None
        view_3604: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_1201, [512, 784, 256]);  getitem_1201 = None
        view_3605: "bf16[512, 200704]" = torch.ops.aten.view.default(view_3604, [512, -1]);  view_3604 = None
        view_3606: "bf16[512, 64, 56, 56]" = torch.ops.aten.view.default(view_3605, [512, 64, 56, 56]);  view_3605 = None
        convolution_backward_87 = torch.ops.aten.convolution_backward.default(view_3603, view_3606, expand_88, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_3603 = view_3606 = expand_88 = None
        getitem_1206: "bf16[64, 64, 3, 3]" = convolution_backward_87[1];  convolution_backward_87 = None
        convert_element_type_282: "f32[64, 64, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_1206, torch.float32);  getitem_1206 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_529 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 840, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_115, 'Y_ptr': full_default_339, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_115 = None
        getitem_1208: "i8[401408, 256]" = triton_kernel_wrapper_functional_proxy_529['Y_ptr'];  triton_kernel_wrapper_functional_proxy_529 = None
        view_3610: "i8[512, 784, 256]" = torch.ops.aten.view.default(getitem_1208, [512, 784, 256]);  getitem_1208 = None
        view_3611: "i8[512, 200704]" = torch.ops.aten.view.default(view_3610, [512, -1]);  view_3610 = None
        view_3612: "i8[512, 64, 56, 56]" = torch.ops.aten.view.default(view_3611, [512, 64, 56, 56]);  view_3611 = None
        mul_412: "bf16[512, 64, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_1202, view_3612);  getitem_1202 = view_3612 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_697: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_530 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 612, constant_args_idx = 841, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_110, 'S_ptr': getitem_111, 'M_ptr': getitem_112, 'Y_ptr': empty_697, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_110 = getitem_111 = getitem_112 = empty_697 = None
        getitem_1209: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_530['Y_ptr'];  triton_kernel_wrapper_functional_proxy_530 = None
        view_3628: "bf16[512, 64, 3136]" = torch.ops.aten.view.default(mul_412, [512, 64, 3136]);  mul_412 = None
        view_3629: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_1209, [512, 784, 256]);  getitem_1209 = None
        view_3630: "bf16[512, 200704]" = torch.ops.aten.view.default(view_3629, [512, -1]);  view_3629 = None
        view_3631: "bf16[512, 64, 3136]" = torch.ops.aten.view.default(view_3630, [512, 64, 3136]);  view_3630 = None
        triton_kernel_wrapper_functional_proxy_531 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 613, constant_args_idx = 842, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3631, 'DY': view_3628, 'DBETA': full_default, 'DGAMMA': full_default, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1210: "f32[64]" = triton_kernel_wrapper_functional_proxy_531['DBETA']
        getitem_1211: "f32[64]" = triton_kernel_wrapper_functional_proxy_531['DGAMMA'];  triton_kernel_wrapper_functional_proxy_531 = None
        empty_698: "bf16[512, 64, 3136]" = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_201: "bf16[512, 64, 3136]" = torch.ops.aten.permute.default(empty_698, [0, 1, 2]);  empty_698 = None
        triton_kernel_wrapper_functional_proxy_532 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 614, constant_args_idx = 843, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3631, 'DY': view_3628, 'INVSTD': rsqrt_8, 'GAMMA': primals_52, 'DBETA': getitem_1210, 'DGAMMA': getitem_1211, 'DX': permute_201, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3631 = view_3628 = rsqrt_8 = primals_52 = permute_201 = None
        getitem_1212: "bf16[512, 64, 3136]" = triton_kernel_wrapper_functional_proxy_532['DX'];  triton_kernel_wrapper_functional_proxy_532 = None
        convert_element_type_default_17: "f32[64]" = torch.ops.prims.convert_element_type.default(getitem_1211, torch.float32);  getitem_1211 = None
        convert_element_type_default_16: "f32[64]" = torch.ops.prims.convert_element_type.default(getitem_1210, torch.float32);  getitem_1210 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_699: "bf16[1605632, 256]" = torch.ops.aten.empty.memory_format([1605632, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_533 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 615, constant_args_idx = 844, grid = [(1605632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_103, 'S_ptr': getitem_104, 'M_ptr': getitem_105, 'Y_ptr': empty_699, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_103 = getitem_104 = getitem_105 = empty_699 = None
        getitem_1213: "bf16[1605632, 256]" = triton_kernel_wrapper_functional_proxy_533['Y_ptr'];  triton_kernel_wrapper_functional_proxy_533 = None
        view_3648: "bf16[512, 64, 56, 56]" = torch.ops.aten.view.default(getitem_1212, [512, 64, 56, 56]);  getitem_1212 = None
        empty_700: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_89: "bf16[512, 256, 56, 56]" = torch.ops.aten.expand.default(empty_700, [512, 256, 56, 56]);  empty_700 = None
        convolution_backward_88 = torch.ops.aten.convolution_backward.default(view_3648, expand_89, convert_element_type_9, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_89 = convert_element_type_9 = None
        getitem_1214: "bf16[512, 256, 56, 56]" = convolution_backward_88[0];  convolution_backward_88 = None
        empty_701: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_90: "bf16[64, 256, 1, 1]" = torch.ops.aten.expand.default(empty_701, [64, 256, 1, 1]);  empty_701 = None
        view_3649: "bf16[512, 3136, 256]" = torch.ops.aten.view.default(getitem_1213, [512, 3136, 256]);  getitem_1213 = None
        view_3650: "bf16[512, 802816]" = torch.ops.aten.view.default(view_3649, [512, -1]);  view_3649 = None
        view_3651: "bf16[512, 256, 56, 56]" = torch.ops.aten.view.default(view_3650, [512, 256, 56, 56]);  view_3650 = None
        convolution_backward_89 = torch.ops.aten.convolution_backward.default(view_3648, view_3651, expand_90, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_3648 = view_3651 = expand_90 = None
        getitem_1218: "bf16[64, 256, 1, 1]" = convolution_backward_89[1];  convolution_backward_89 = None
        convert_element_type_287: "f32[64, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1218, torch.float32);  getitem_1218 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_241: "bf16[512, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_410, getitem_1214);  mul_410 = getitem_1214 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_534 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 845, grid = [(1605632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_102, 'Y_ptr': full_default_433, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_102 = None
        getitem_1220: "i8[1605632, 256]" = triton_kernel_wrapper_functional_proxy_534['Y_ptr'];  triton_kernel_wrapper_functional_proxy_534 = None
        view_3655: "i8[512, 3136, 256]" = torch.ops.aten.view.default(getitem_1220, [512, 3136, 256]);  getitem_1220 = None
        view_3656: "i8[512, 802816]" = torch.ops.aten.view.default(view_3655, [512, -1]);  view_3655 = None
        view_3657: "i8[512, 256, 56, 56]" = torch.ops.aten.view.default(view_3656, [512, 256, 56, 56]);  view_3656 = None
        mul_413: "bf16[512, 256, 56, 56]" = torch.ops.aten.mul.Tensor(add_241, view_3657);  add_241 = view_3657 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_702: "bf16[1605632, 256]" = torch.ops.aten.empty.memory_format([1605632, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_535 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 616, constant_args_idx = 846, grid = [(1605632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_97, 'S_ptr': getitem_98, 'M_ptr': getitem_99, 'Y_ptr': empty_702, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_97 = getitem_98 = getitem_99 = empty_702 = None
        getitem_1221: "bf16[1605632, 256]" = triton_kernel_wrapper_functional_proxy_535['Y_ptr'];  triton_kernel_wrapper_functional_proxy_535 = None
        view_3673: "bf16[512, 256, 3136]" = torch.ops.aten.view.default(mul_413, [512, 256, 3136])
        view_3674: "bf16[512, 3136, 256]" = torch.ops.aten.view.default(getitem_1221, [512, 3136, 256]);  getitem_1221 = None
        view_3675: "bf16[512, 802816]" = torch.ops.aten.view.default(view_3674, [512, -1]);  view_3674 = None
        view_3676: "bf16[512, 256, 3136]" = torch.ops.aten.view.default(view_3675, [512, 256, 3136]);  view_3675 = None
        triton_kernel_wrapper_functional_proxy_536 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 617, constant_args_idx = 847, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3676, 'DY': view_3673, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1222: "f32[256]" = triton_kernel_wrapper_functional_proxy_536['DBETA']
        getitem_1223: "f32[256]" = triton_kernel_wrapper_functional_proxy_536['DGAMMA'];  triton_kernel_wrapper_functional_proxy_536 = None
        empty_703: "bf16[512, 256, 3136]" = torch.ops.aten.empty.memory_format([512, 256, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_202: "bf16[512, 256, 3136]" = torch.ops.aten.permute.default(empty_703, [0, 1, 2]);  empty_703 = None
        triton_kernel_wrapper_functional_proxy_537 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 618, constant_args_idx = 848, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3676, 'DY': view_3673, 'INVSTD': rsqrt_7, 'GAMMA': primals_46, 'DBETA': getitem_1222, 'DGAMMA': getitem_1223, 'DX': permute_202, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3676 = view_3673 = rsqrt_7 = primals_46 = permute_202 = None
        getitem_1224: "bf16[512, 256, 3136]" = triton_kernel_wrapper_functional_proxy_537['DX'];  triton_kernel_wrapper_functional_proxy_537 = None
        convert_element_type_default_15: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_1223, torch.float32);  getitem_1223 = None
        convert_element_type_default_14: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_1222, torch.float32);  getitem_1222 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_704: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_538 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 619, constant_args_idx = 849, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_90, 'S_ptr': getitem_91, 'M_ptr': getitem_92, 'Y_ptr': empty_704, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_90 = getitem_91 = getitem_92 = empty_704 = None
        getitem_1225: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_538['Y_ptr'];  triton_kernel_wrapper_functional_proxy_538 = None
        view_3693: "bf16[512, 256, 56, 56]" = torch.ops.aten.view.default(getitem_1224, [512, 256, 56, 56]);  getitem_1224 = None
        empty_705: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_91: "bf16[512, 64, 56, 56]" = torch.ops.aten.expand.default(empty_705, [512, 64, 56, 56]);  empty_705 = None
        convolution_backward_90 = torch.ops.aten.convolution_backward.default(view_3693, expand_91, convert_element_type_8, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_91 = convert_element_type_8 = None
        getitem_1226: "bf16[512, 64, 56, 56]" = convolution_backward_90[0];  convolution_backward_90 = None
        empty_706: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_92: "bf16[256, 64, 1, 1]" = torch.ops.aten.expand.default(empty_706, [256, 64, 1, 1]);  empty_706 = None
        view_3694: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_1225, [512, 784, 256]);  getitem_1225 = None
        view_3695: "bf16[512, 200704]" = torch.ops.aten.view.default(view_3694, [512, -1]);  view_3694 = None
        view_3696: "bf16[512, 64, 56, 56]" = torch.ops.aten.view.default(view_3695, [512, 64, 56, 56]);  view_3695 = None
        convolution_backward_91 = torch.ops.aten.convolution_backward.default(view_3693, view_3696, expand_92, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_3693 = view_3696 = expand_92 = None
        getitem_1230: "bf16[256, 64, 1, 1]" = convolution_backward_91[1];  convolution_backward_91 = None
        convert_element_type_292: "f32[256, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1230, torch.float32);  getitem_1230 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_539 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 850, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_89, 'Y_ptr': full_default_339, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_89 = None
        getitem_1232: "i8[401408, 256]" = triton_kernel_wrapper_functional_proxy_539['Y_ptr'];  triton_kernel_wrapper_functional_proxy_539 = None
        view_3700: "i8[512, 784, 256]" = torch.ops.aten.view.default(getitem_1232, [512, 784, 256]);  getitem_1232 = None
        view_3701: "i8[512, 200704]" = torch.ops.aten.view.default(view_3700, [512, -1]);  view_3700 = None
        view_3702: "i8[512, 64, 56, 56]" = torch.ops.aten.view.default(view_3701, [512, 64, 56, 56]);  view_3701 = None
        mul_414: "bf16[512, 64, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_1226, view_3702);  getitem_1226 = view_3702 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_707: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_540 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 620, constant_args_idx = 851, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_84, 'S_ptr': getitem_85, 'M_ptr': getitem_86, 'Y_ptr': empty_707, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_84 = getitem_85 = getitem_86 = empty_707 = None
        getitem_1233: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_540['Y_ptr'];  triton_kernel_wrapper_functional_proxy_540 = None
        view_3718: "bf16[512, 64, 3136]" = torch.ops.aten.view.default(mul_414, [512, 64, 3136]);  mul_414 = None
        view_3719: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_1233, [512, 784, 256]);  getitem_1233 = None
        view_3720: "bf16[512, 200704]" = torch.ops.aten.view.default(view_3719, [512, -1]);  view_3719 = None
        view_3721: "bf16[512, 64, 3136]" = torch.ops.aten.view.default(view_3720, [512, 64, 3136]);  view_3720 = None
        triton_kernel_wrapper_functional_proxy_541 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 621, constant_args_idx = 852, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3721, 'DY': view_3718, 'DBETA': full_default, 'DGAMMA': full_default, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1234: "f32[64]" = triton_kernel_wrapper_functional_proxy_541['DBETA']
        getitem_1235: "f32[64]" = triton_kernel_wrapper_functional_proxy_541['DGAMMA'];  triton_kernel_wrapper_functional_proxy_541 = None
        empty_708: "bf16[512, 64, 3136]" = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_203: "bf16[512, 64, 3136]" = torch.ops.aten.permute.default(empty_708, [0, 1, 2]);  empty_708 = None
        triton_kernel_wrapper_functional_proxy_542 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 622, constant_args_idx = 853, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3721, 'DY': view_3718, 'INVSTD': rsqrt_6, 'GAMMA': primals_40, 'DBETA': getitem_1234, 'DGAMMA': getitem_1235, 'DX': permute_203, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3721 = view_3718 = rsqrt_6 = primals_40 = permute_203 = None
        getitem_1236: "bf16[512, 64, 3136]" = triton_kernel_wrapper_functional_proxy_542['DX'];  triton_kernel_wrapper_functional_proxy_542 = None
        convert_element_type_default_13: "f32[64]" = torch.ops.prims.convert_element_type.default(getitem_1235, torch.float32);  getitem_1235 = None
        convert_element_type_default_12: "f32[64]" = torch.ops.prims.convert_element_type.default(getitem_1234, torch.float32);  getitem_1234 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_709: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_543 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 623, constant_args_idx = 854, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_77, 'S_ptr': getitem_78, 'M_ptr': getitem_79, 'Y_ptr': empty_709, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_77 = getitem_78 = getitem_79 = empty_709 = None
        getitem_1237: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_543['Y_ptr'];  triton_kernel_wrapper_functional_proxy_543 = None
        view_3738: "bf16[512, 64, 56, 56]" = torch.ops.aten.view.default(getitem_1236, [512, 64, 56, 56]);  getitem_1236 = None
        empty_710: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_93: "bf16[512, 64, 56, 56]" = torch.ops.aten.expand.default(empty_710, [512, 64, 56, 56]);  empty_710 = None
        convolution_backward_92 = torch.ops.aten.convolution_backward.default(view_3738, expand_93, convert_element_type_7, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_93 = convert_element_type_7 = None
        getitem_1238: "bf16[512, 64, 56, 56]" = convolution_backward_92[0];  convolution_backward_92 = None
        empty_711: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_94: "bf16[64, 64, 3, 3]" = torch.ops.aten.expand.default(empty_711, [64, 64, 3, 3]);  empty_711 = None
        view_3739: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_1237, [512, 784, 256]);  getitem_1237 = None
        view_3740: "bf16[512, 200704]" = torch.ops.aten.view.default(view_3739, [512, -1]);  view_3739 = None
        view_3741: "bf16[512, 64, 56, 56]" = torch.ops.aten.view.default(view_3740, [512, 64, 56, 56]);  view_3740 = None
        convolution_backward_93 = torch.ops.aten.convolution_backward.default(view_3738, view_3741, expand_94, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_3738 = view_3741 = expand_94 = None
        getitem_1242: "bf16[64, 64, 3, 3]" = convolution_backward_93[1];  convolution_backward_93 = None
        convert_element_type_297: "f32[64, 64, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_1242, torch.float32);  getitem_1242 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_544 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 855, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_76, 'Y_ptr': full_default_339, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_76 = None
        getitem_1244: "i8[401408, 256]" = triton_kernel_wrapper_functional_proxy_544['Y_ptr'];  triton_kernel_wrapper_functional_proxy_544 = None
        view_3745: "i8[512, 784, 256]" = torch.ops.aten.view.default(getitem_1244, [512, 784, 256]);  getitem_1244 = None
        view_3746: "i8[512, 200704]" = torch.ops.aten.view.default(view_3745, [512, -1]);  view_3745 = None
        view_3747: "i8[512, 64, 56, 56]" = torch.ops.aten.view.default(view_3746, [512, 64, 56, 56]);  view_3746 = None
        mul_415: "bf16[512, 64, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_1238, view_3747);  getitem_1238 = view_3747 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_712: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_545 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 624, constant_args_idx = 856, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_71, 'S_ptr': getitem_72, 'M_ptr': getitem_73, 'Y_ptr': empty_712, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_71 = getitem_72 = getitem_73 = empty_712 = None
        getitem_1245: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_545['Y_ptr'];  triton_kernel_wrapper_functional_proxy_545 = None
        view_3763: "bf16[512, 64, 3136]" = torch.ops.aten.view.default(mul_415, [512, 64, 3136]);  mul_415 = None
        view_3764: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_1245, [512, 784, 256]);  getitem_1245 = None
        view_3765: "bf16[512, 200704]" = torch.ops.aten.view.default(view_3764, [512, -1]);  view_3764 = None
        view_3766: "bf16[512, 64, 3136]" = torch.ops.aten.view.default(view_3765, [512, 64, 3136]);  view_3765 = None
        triton_kernel_wrapper_functional_proxy_546 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 625, constant_args_idx = 857, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3766, 'DY': view_3763, 'DBETA': full_default, 'DGAMMA': full_default, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1246: "f32[64]" = triton_kernel_wrapper_functional_proxy_546['DBETA']
        getitem_1247: "f32[64]" = triton_kernel_wrapper_functional_proxy_546['DGAMMA'];  triton_kernel_wrapper_functional_proxy_546 = None
        empty_713: "bf16[512, 64, 3136]" = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_204: "bf16[512, 64, 3136]" = torch.ops.aten.permute.default(empty_713, [0, 1, 2]);  empty_713 = None
        triton_kernel_wrapper_functional_proxy_547 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 626, constant_args_idx = 858, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3766, 'DY': view_3763, 'INVSTD': rsqrt_5, 'GAMMA': primals_34, 'DBETA': getitem_1246, 'DGAMMA': getitem_1247, 'DX': permute_204, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3766 = view_3763 = rsqrt_5 = primals_34 = permute_204 = None
        getitem_1248: "bf16[512, 64, 3136]" = triton_kernel_wrapper_functional_proxy_547['DX'];  triton_kernel_wrapper_functional_proxy_547 = None
        convert_element_type_default_11: "f32[64]" = torch.ops.prims.convert_element_type.default(getitem_1247, torch.float32);  getitem_1247 = None
        convert_element_type_default_10: "f32[64]" = torch.ops.prims.convert_element_type.default(getitem_1246, torch.float32);  getitem_1246 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_714: "bf16[1605632, 256]" = torch.ops.aten.empty.memory_format([1605632, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_548 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 627, constant_args_idx = 859, grid = [(1605632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_64, 'S_ptr': getitem_65, 'M_ptr': getitem_66, 'Y_ptr': empty_714, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_64 = getitem_65 = getitem_66 = empty_714 = None
        getitem_1249: "bf16[1605632, 256]" = triton_kernel_wrapper_functional_proxy_548['Y_ptr'];  triton_kernel_wrapper_functional_proxy_548 = None
        view_3783: "bf16[512, 64, 56, 56]" = torch.ops.aten.view.default(getitem_1248, [512, 64, 56, 56]);  getitem_1248 = None
        empty_715: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_95: "bf16[512, 256, 56, 56]" = torch.ops.aten.expand.default(empty_715, [512, 256, 56, 56]);  empty_715 = None
        convolution_backward_94 = torch.ops.aten.convolution_backward.default(view_3783, expand_95, convert_element_type_6, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_95 = convert_element_type_6 = None
        getitem_1250: "bf16[512, 256, 56, 56]" = convolution_backward_94[0];  convolution_backward_94 = None
        empty_716: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_96: "bf16[64, 256, 1, 1]" = torch.ops.aten.expand.default(empty_716, [64, 256, 1, 1]);  empty_716 = None
        view_3784: "bf16[512, 3136, 256]" = torch.ops.aten.view.default(getitem_1249, [512, 3136, 256]);  getitem_1249 = None
        view_3785: "bf16[512, 802816]" = torch.ops.aten.view.default(view_3784, [512, -1]);  view_3784 = None
        view_3786: "bf16[512, 256, 56, 56]" = torch.ops.aten.view.default(view_3785, [512, 256, 56, 56]);  view_3785 = None
        convolution_backward_95 = torch.ops.aten.convolution_backward.default(view_3783, view_3786, expand_96, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_3783 = view_3786 = expand_96 = None
        getitem_1254: "bf16[64, 256, 1, 1]" = convolution_backward_95[1];  convolution_backward_95 = None
        convert_element_type_302: "f32[64, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1254, torch.float32);  getitem_1254 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_242: "bf16[512, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_413, getitem_1250);  mul_413 = getitem_1250 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_549 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 860, grid = [(1605632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_63, 'Y_ptr': full_default_433, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_63 = None
        getitem_1256: "i8[1605632, 256]" = triton_kernel_wrapper_functional_proxy_549['Y_ptr'];  triton_kernel_wrapper_functional_proxy_549 = None
        view_3790: "i8[512, 3136, 256]" = torch.ops.aten.view.default(getitem_1256, [512, 3136, 256]);  getitem_1256 = None
        view_3791: "i8[512, 802816]" = torch.ops.aten.view.default(view_3790, [512, -1]);  view_3790 = None
        view_3792: "i8[512, 256, 56, 56]" = torch.ops.aten.view.default(view_3791, [512, 256, 56, 56]);  view_3791 = None
        mul_416: "bf16[512, 256, 56, 56]" = torch.ops.aten.mul.Tensor(add_242, view_3792);  add_242 = view_3792 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_717: "bf16[1605632, 256]" = torch.ops.aten.empty.memory_format([1605632, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_550 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 628, constant_args_idx = 861, grid = [(1605632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_58, 'S_ptr': getitem_59, 'M_ptr': getitem_60, 'Y_ptr': empty_717, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_58 = getitem_59 = getitem_60 = empty_717 = None
        getitem_1257: "bf16[1605632, 256]" = triton_kernel_wrapper_functional_proxy_550['Y_ptr'];  triton_kernel_wrapper_functional_proxy_550 = None
        view_3808: "bf16[512, 256, 3136]" = torch.ops.aten.view.default(mul_416, [512, 256, 3136]);  mul_416 = None
        view_3809: "bf16[512, 3136, 256]" = torch.ops.aten.view.default(getitem_1257, [512, 3136, 256]);  getitem_1257 = None
        view_3810: "bf16[512, 802816]" = torch.ops.aten.view.default(view_3809, [512, -1]);  view_3809 = None
        view_3811: "bf16[512, 256, 3136]" = torch.ops.aten.view.default(view_3810, [512, 256, 3136]);  view_3810 = None
        triton_kernel_wrapper_functional_proxy_551 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 629, constant_args_idx = 862, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3811, 'DY': view_3808, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1258: "f32[256]" = triton_kernel_wrapper_functional_proxy_551['DBETA']
        getitem_1259: "f32[256]" = triton_kernel_wrapper_functional_proxy_551['DGAMMA'];  triton_kernel_wrapper_functional_proxy_551 = None
        empty_718: "bf16[512, 256, 3136]" = torch.ops.aten.empty.memory_format([512, 256, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_205: "bf16[512, 256, 3136]" = torch.ops.aten.permute.default(empty_718, [0, 1, 2]);  empty_718 = None
        triton_kernel_wrapper_functional_proxy_552 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 630, constant_args_idx = 863, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3811, 'DY': view_3808, 'INVSTD': rsqrt_4, 'GAMMA': primals_28, 'DBETA': getitem_1258, 'DGAMMA': getitem_1259, 'DX': permute_205, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3811 = rsqrt_4 = primals_28 = permute_205 = None
        getitem_1260: "bf16[512, 256, 3136]" = triton_kernel_wrapper_functional_proxy_552['DX'];  triton_kernel_wrapper_functional_proxy_552 = None
        convert_element_type_default_9: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_1259, torch.float32);  getitem_1259 = None
        convert_element_type_default_8: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_1258, torch.float32);  getitem_1258 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_719: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_553 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 631, constant_args_idx = 864, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_51, 'S_ptr': getitem_52, 'M_ptr': getitem_53, 'Y_ptr': empty_719, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_51 = getitem_52 = getitem_53 = empty_719 = None
        getitem_1261: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_553['Y_ptr'];  triton_kernel_wrapper_functional_proxy_553 = None
        view_3828: "bf16[512, 256, 56, 56]" = torch.ops.aten.view.default(getitem_1260, [512, 256, 56, 56]);  getitem_1260 = None
        empty_720: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_97: "bf16[512, 64, 56, 56]" = torch.ops.aten.expand.default(empty_720, [512, 64, 56, 56]);  empty_720 = None
        convolution_backward_96 = torch.ops.aten.convolution_backward.default(view_3828, expand_97, convert_element_type_5, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_97 = convert_element_type_5 = None
        getitem_1262: "bf16[512, 64, 56, 56]" = convolution_backward_96[0];  convolution_backward_96 = None
        empty_721: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_98: "bf16[256, 64, 1, 1]" = torch.ops.aten.expand.default(empty_721, [256, 64, 1, 1]);  empty_721 = None
        view_3829: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_1261, [512, 784, 256]);  getitem_1261 = None
        view_3830: "bf16[512, 200704]" = torch.ops.aten.view.default(view_3829, [512, -1]);  view_3829 = None
        view_3831: "bf16[512, 64, 56, 56]" = torch.ops.aten.view.default(view_3830, [512, 64, 56, 56]);  view_3830 = None
        convolution_backward_97 = torch.ops.aten.convolution_backward.default(view_3828, view_3831, expand_98, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_3828 = view_3831 = expand_98 = None
        getitem_1266: "bf16[256, 64, 1, 1]" = convolution_backward_97[1];  convolution_backward_97 = None
        convert_element_type_307: "f32[256, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1266, torch.float32);  getitem_1266 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_722: "bf16[1605632, 256]" = torch.ops.aten.empty.memory_format([1605632, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_554 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 632, constant_args_idx = 865, grid = [(1605632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_48, 'S_ptr': getitem_49, 'M_ptr': getitem_50, 'Y_ptr': empty_722, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_48 = getitem_49 = getitem_50 = empty_722 = None
        getitem_1268: "bf16[1605632, 256]" = triton_kernel_wrapper_functional_proxy_554['Y_ptr'];  triton_kernel_wrapper_functional_proxy_554 = None
        view_3848: "bf16[512, 3136, 256]" = torch.ops.aten.view.default(getitem_1268, [512, 3136, 256]);  getitem_1268 = None
        view_3849: "bf16[512, 802816]" = torch.ops.aten.view.default(view_3848, [512, -1]);  view_3848 = None
        view_3850: "bf16[512, 256, 3136]" = torch.ops.aten.view.default(view_3849, [512, 256, 3136]);  view_3849 = None
        triton_kernel_wrapper_functional_proxy_555 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 633, constant_args_idx = 866, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3850, 'DY': view_3808, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA']);  full_default_18 = None
        getitem_1269: "f32[256]" = triton_kernel_wrapper_functional_proxy_555['DBETA']
        getitem_1270: "f32[256]" = triton_kernel_wrapper_functional_proxy_555['DGAMMA'];  triton_kernel_wrapper_functional_proxy_555 = None
        empty_723: "bf16[512, 256, 3136]" = torch.ops.aten.empty.memory_format([512, 256, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_206: "bf16[512, 256, 3136]" = torch.ops.aten.permute.default(empty_723, [0, 1, 2]);  empty_723 = None
        triton_kernel_wrapper_functional_proxy_556 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 634, constant_args_idx = 867, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3850, 'DY': view_3808, 'INVSTD': rsqrt_3, 'GAMMA': primals_22, 'DBETA': getitem_1269, 'DGAMMA': getitem_1270, 'DX': permute_206, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3850 = view_3808 = rsqrt_3 = primals_22 = permute_206 = None
        getitem_1271: "bf16[512, 256, 3136]" = triton_kernel_wrapper_functional_proxy_556['DX'];  triton_kernel_wrapper_functional_proxy_556 = None
        convert_element_type_default_7: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_1270, torch.float32);  getitem_1270 = None
        convert_element_type_default_6: "f32[256]" = torch.ops.prims.convert_element_type.default(getitem_1269, torch.float32);  getitem_1269 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_724: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_557 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 635, constant_args_idx = 868, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_41, 'S_ptr': getitem_42, 'M_ptr': getitem_43, 'Y_ptr': empty_724, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_41 = getitem_42 = getitem_43 = empty_724 = None
        getitem_1272: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_557['Y_ptr'];  triton_kernel_wrapper_functional_proxy_557 = None
        view_3867: "bf16[512, 256, 56, 56]" = torch.ops.aten.view.default(getitem_1271, [512, 256, 56, 56]);  getitem_1271 = None
        empty_725: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_99: "bf16[512, 64, 56, 56]" = torch.ops.aten.expand.default(empty_725, [512, 64, 56, 56]);  empty_725 = None
        convolution_backward_98 = torch.ops.aten.convolution_backward.default(view_3867, expand_99, convert_element_type_4, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_99 = convert_element_type_4 = None
        getitem_1273: "bf16[512, 64, 56, 56]" = convolution_backward_98[0];  convolution_backward_98 = None
        empty_726: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_100: "bf16[256, 64, 1, 1]" = torch.ops.aten.expand.default(empty_726, [256, 64, 1, 1]);  empty_726 = None
        view_3868: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_1272, [512, 784, 256]);  getitem_1272 = None
        view_3869: "bf16[512, 200704]" = torch.ops.aten.view.default(view_3868, [512, -1]);  view_3868 = None
        view_3870: "bf16[512, 64, 56, 56]" = torch.ops.aten.view.default(view_3869, [512, 64, 56, 56]);  view_3869 = None
        convolution_backward_99 = torch.ops.aten.convolution_backward.default(view_3867, view_3870, expand_100, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_3867 = view_3870 = expand_100 = None
        getitem_1277: "bf16[256, 64, 1, 1]" = convolution_backward_99[1];  convolution_backward_99 = None
        convert_element_type_312: "f32[256, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1277, torch.float32);  getitem_1277 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_558 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 869, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_40, 'Y_ptr': full_default_339, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_40 = None
        getitem_1279: "i8[401408, 256]" = triton_kernel_wrapper_functional_proxy_558['Y_ptr'];  triton_kernel_wrapper_functional_proxy_558 = None
        view_3874: "i8[512, 784, 256]" = torch.ops.aten.view.default(getitem_1279, [512, 784, 256]);  getitem_1279 = None
        view_3875: "i8[512, 200704]" = torch.ops.aten.view.default(view_3874, [512, -1]);  view_3874 = None
        view_3876: "i8[512, 64, 56, 56]" = torch.ops.aten.view.default(view_3875, [512, 64, 56, 56]);  view_3875 = None
        mul_417: "bf16[512, 64, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_1273, view_3876);  getitem_1273 = view_3876 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_727: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_559 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 636, constant_args_idx = 870, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_35, 'S_ptr': getitem_36, 'M_ptr': getitem_37, 'Y_ptr': empty_727, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_35 = getitem_36 = getitem_37 = empty_727 = None
        getitem_1280: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_559['Y_ptr'];  triton_kernel_wrapper_functional_proxy_559 = None
        view_3892: "bf16[512, 64, 3136]" = torch.ops.aten.view.default(mul_417, [512, 64, 3136]);  mul_417 = None
        view_3893: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_1280, [512, 784, 256]);  getitem_1280 = None
        view_3894: "bf16[512, 200704]" = torch.ops.aten.view.default(view_3893, [512, -1]);  view_3893 = None
        view_3895: "bf16[512, 64, 3136]" = torch.ops.aten.view.default(view_3894, [512, 64, 3136]);  view_3894 = None
        triton_kernel_wrapper_functional_proxy_560 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 637, constant_args_idx = 871, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3895, 'DY': view_3892, 'DBETA': full_default, 'DGAMMA': full_default, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1281: "f32[64]" = triton_kernel_wrapper_functional_proxy_560['DBETA']
        getitem_1282: "f32[64]" = triton_kernel_wrapper_functional_proxy_560['DGAMMA'];  triton_kernel_wrapper_functional_proxy_560 = None
        empty_728: "bf16[512, 64, 3136]" = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_207: "bf16[512, 64, 3136]" = torch.ops.aten.permute.default(empty_728, [0, 1, 2]);  empty_728 = None
        triton_kernel_wrapper_functional_proxy_561 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 638, constant_args_idx = 872, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3895, 'DY': view_3892, 'INVSTD': rsqrt_2, 'GAMMA': primals_16, 'DBETA': getitem_1281, 'DGAMMA': getitem_1282, 'DX': permute_207, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3895 = view_3892 = rsqrt_2 = primals_16 = permute_207 = None
        getitem_1283: "bf16[512, 64, 3136]" = triton_kernel_wrapper_functional_proxy_561['DX'];  triton_kernel_wrapper_functional_proxy_561 = None
        convert_element_type_default_5: "f32[64]" = torch.ops.prims.convert_element_type.default(getitem_1282, torch.float32);  getitem_1282 = None
        convert_element_type_default_4: "f32[64]" = torch.ops.prims.convert_element_type.default(getitem_1281, torch.float32);  getitem_1281 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_729: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_562 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 639, constant_args_idx = 873, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_28, 'S_ptr': getitem_29, 'M_ptr': getitem_30, 'Y_ptr': empty_729, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_28 = getitem_29 = getitem_30 = empty_729 = None
        getitem_1284: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_562['Y_ptr'];  triton_kernel_wrapper_functional_proxy_562 = None
        view_3912: "bf16[512, 64, 56, 56]" = torch.ops.aten.view.default(getitem_1283, [512, 64, 56, 56]);  getitem_1283 = None
        empty_730: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_101: "bf16[512, 64, 56, 56]" = torch.ops.aten.expand.default(empty_730, [512, 64, 56, 56]);  empty_730 = None
        convolution_backward_100 = torch.ops.aten.convolution_backward.default(view_3912, expand_101, convert_element_type_3, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_101 = convert_element_type_3 = None
        getitem_1285: "bf16[512, 64, 56, 56]" = convolution_backward_100[0];  convolution_backward_100 = None
        empty_731: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_102: "bf16[64, 64, 3, 3]" = torch.ops.aten.expand.default(empty_731, [64, 64, 3, 3]);  empty_731 = None
        view_3913: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_1284, [512, 784, 256]);  getitem_1284 = None
        view_3914: "bf16[512, 200704]" = torch.ops.aten.view.default(view_3913, [512, -1]);  view_3913 = None
        view_3915: "bf16[512, 64, 56, 56]" = torch.ops.aten.view.default(view_3914, [512, 64, 56, 56]);  view_3914 = None
        convolution_backward_101 = torch.ops.aten.convolution_backward.default(view_3912, view_3915, expand_102, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_3912 = view_3915 = expand_102 = None
        getitem_1289: "bf16[64, 64, 3, 3]" = convolution_backward_101[1];  convolution_backward_101 = None
        convert_element_type_317: "f32[64, 64, 3, 3]" = torch.ops.prims.convert_element_type.default(getitem_1289, torch.float32);  getitem_1289 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_563 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 874, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_27, 'Y_ptr': full_default_339, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_27 = full_default_339 = None
        getitem_1291: "i8[401408, 256]" = triton_kernel_wrapper_functional_proxy_563['Y_ptr'];  triton_kernel_wrapper_functional_proxy_563 = None
        view_3919: "i8[512, 784, 256]" = torch.ops.aten.view.default(getitem_1291, [512, 784, 256]);  getitem_1291 = None
        view_3920: "i8[512, 200704]" = torch.ops.aten.view.default(view_3919, [512, -1]);  view_3919 = None
        view_3921: "i8[512, 64, 56, 56]" = torch.ops.aten.view.default(view_3920, [512, 64, 56, 56]);  view_3920 = None
        mul_418: "bf16[512, 64, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_1285, view_3921);  getitem_1285 = view_3921 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_732: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_564 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 640, constant_args_idx = 875, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_22, 'S_ptr': getitem_23, 'M_ptr': getitem_24, 'Y_ptr': empty_732, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_22 = getitem_23 = getitem_24 = empty_732 = None
        getitem_1292: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_564['Y_ptr'];  triton_kernel_wrapper_functional_proxy_564 = None
        view_3937: "bf16[512, 64, 3136]" = torch.ops.aten.view.default(mul_418, [512, 64, 3136]);  mul_418 = None
        view_3938: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_1292, [512, 784, 256]);  getitem_1292 = None
        view_3939: "bf16[512, 200704]" = torch.ops.aten.view.default(view_3938, [512, -1]);  view_3938 = None
        view_3940: "bf16[512, 64, 3136]" = torch.ops.aten.view.default(view_3939, [512, 64, 3136]);  view_3939 = None
        triton_kernel_wrapper_functional_proxy_565 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 641, constant_args_idx = 876, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3940, 'DY': view_3937, 'DBETA': full_default, 'DGAMMA': full_default, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1293: "f32[64]" = triton_kernel_wrapper_functional_proxy_565['DBETA']
        getitem_1294: "f32[64]" = triton_kernel_wrapper_functional_proxy_565['DGAMMA'];  triton_kernel_wrapper_functional_proxy_565 = None
        empty_733: "bf16[512, 64, 3136]" = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_208: "bf16[512, 64, 3136]" = torch.ops.aten.permute.default(empty_733, [0, 1, 2]);  empty_733 = None
        triton_kernel_wrapper_functional_proxy_566 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 642, constant_args_idx = 877, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3940, 'DY': view_3937, 'INVSTD': rsqrt_1, 'GAMMA': primals_10, 'DBETA': getitem_1293, 'DGAMMA': getitem_1294, 'DX': permute_208, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3940 = view_3937 = rsqrt_1 = primals_10 = permute_208 = None
        getitem_1295: "bf16[512, 64, 3136]" = triton_kernel_wrapper_functional_proxy_566['DX'];  triton_kernel_wrapper_functional_proxy_566 = None
        convert_element_type_default_3: "f32[64]" = torch.ops.prims.convert_element_type.default(getitem_1294, torch.float32);  getitem_1294 = None
        convert_element_type_default_2: "f32[64]" = torch.ops.prims.convert_element_type.default(getitem_1293, torch.float32);  getitem_1293 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_734: "bf16[401408, 256]" = torch.ops.aten.empty.memory_format([401408, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_567 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 643, constant_args_idx = 878, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_15, 'S_ptr': getitem_16, 'M_ptr': getitem_17, 'Y_ptr': empty_734, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_15 = getitem_16 = getitem_17 = empty_734 = None
        getitem_1296: "bf16[401408, 256]" = triton_kernel_wrapper_functional_proxy_567['Y_ptr'];  triton_kernel_wrapper_functional_proxy_567 = None
        view_3957: "bf16[512, 64, 56, 56]" = torch.ops.aten.view.default(getitem_1295, [512, 64, 56, 56]);  getitem_1295 = None
        empty_735: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_103: "bf16[512, 64, 56, 56]" = torch.ops.aten.expand.default(empty_735, [512, 64, 56, 56]);  empty_735 = None
        convolution_backward_102 = torch.ops.aten.convolution_backward.default(view_3957, expand_103, convert_element_type_2, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_103 = convert_element_type_2 = None
        getitem_1297: "bf16[512, 64, 56, 56]" = convolution_backward_102[0];  convolution_backward_102 = None
        empty_736: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_104: "bf16[64, 64, 1, 1]" = torch.ops.aten.expand.default(empty_736, [64, 64, 1, 1]);  empty_736 = None
        view_3958: "bf16[512, 784, 256]" = torch.ops.aten.view.default(getitem_1296, [512, 784, 256]);  getitem_1296 = None
        view_3959: "bf16[512, 200704]" = torch.ops.aten.view.default(view_3958, [512, -1]);  view_3958 = None
        view_3960: "bf16[512, 64, 56, 56]" = torch.ops.aten.view.default(view_3959, [512, 64, 56, 56]);  view_3959 = None
        convolution_backward_103 = torch.ops.aten.convolution_backward.default(view_3957, view_3960, expand_104, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_3957 = view_3960 = expand_104 = None
        getitem_1301: "bf16[64, 64, 1, 1]" = convolution_backward_103[1];  convolution_backward_103 = None
        convert_element_type_322: "f32[64, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1301, torch.float32);  getitem_1301 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        add_243: "bf16[512, 64, 56, 56]" = torch.ops.aten.add.Tensor(getitem_1262, getitem_1297);  getitem_1262 = getitem_1297 = None
        
         # File: <eval_with_key>.3 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:8 in forward, code: maxpool = self.maxpool(relu);  relu = None
        _low_memory_max_pool_offsets_to_indices: "i64[512, 64, 56, 56]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_14, [3, 3], [112, 112], [2, 2], [1, 1], [1, 1]);  getitem_14 = None
        max_pool2d_with_indices_backward: "bf16[512, 64, 112, 112]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_243, getitem_10, [3, 3], [2, 2], [1, 1], [1, 1], False, _low_memory_max_pool_offsets_to_indices);  add_243 = getitem_10 = _low_memory_max_pool_offsets_to_indices = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:222 in forward, code: return _QuanReLU.apply(x, self.quantizer, self.target_name, self.graph_mode, self.meta)
        triton_kernel_wrapper_functional_proxy_568 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 879, grid = [(1605632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_12, 'Y_ptr': full_default_433, 'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, tensors_to_clone = ['Y_ptr']);  getitem_12 = full_default_433 = None
        getitem_1303: "i8[1605632, 256]" = triton_kernel_wrapper_functional_proxy_568['Y_ptr'];  triton_kernel_wrapper_functional_proxy_568 = None
        view_3964: "i8[512, 3136, 256]" = torch.ops.aten.view.default(getitem_1303, [512, 3136, 256]);  getitem_1303 = None
        view_3965: "i8[512, 802816]" = torch.ops.aten.view.default(view_3964, [512, -1]);  view_3964 = None
        view_3966: "i8[512, 64, 112, 112]" = torch.ops.aten.view.default(view_3965, [512, 64, 112, 112]);  view_3965 = None
        mul_419: "bf16[512, 64, 112, 112]" = torch.ops.aten.mul.Tensor(max_pool2d_with_indices_backward, view_3966);  max_pool2d_with_indices_backward = view_3966 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:198 in forward, code: return _QuanBatchNorm.apply(
        empty_737: "bf16[1605632, 256]" = torch.ops.aten.empty.memory_format([1605632, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_569 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 644, constant_args_idx = 880, grid = [(1605632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_7, 'S_ptr': getitem_8, 'M_ptr': getitem_9, 'Y_ptr': empty_737, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_7 = getitem_8 = getitem_9 = empty_737 = None
        getitem_1304: "bf16[1605632, 256]" = triton_kernel_wrapper_functional_proxy_569['Y_ptr'];  triton_kernel_wrapper_functional_proxy_569 = None
        view_3982: "bf16[512, 64, 12544]" = torch.ops.aten.view.default(mul_419, [512, 64, 12544]);  mul_419 = None
        view_3983: "bf16[512, 3136, 256]" = torch.ops.aten.view.default(getitem_1304, [512, 3136, 256]);  getitem_1304 = None
        view_3984: "bf16[512, 802816]" = torch.ops.aten.view.default(view_3983, [512, -1]);  view_3983 = None
        view_3985: "bf16[512, 64, 12544]" = torch.ops.aten.view.default(view_3984, [512, 64, 12544]);  view_3984 = None
        triton_kernel_wrapper_functional_proxy_570 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 645, constant_args_idx = 881, grid = [(64, 6272, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3985, 'DY': view_3982, 'DBETA': full_default, 'DGAMMA': full_default, 'M': 6422528, 'HW': 12544, 'stride_n': 802816, 'stride_c': 12544, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA']);  full_default = None
        getitem_1305: "f32[64]" = triton_kernel_wrapper_functional_proxy_570['DBETA']
        getitem_1306: "f32[64]" = triton_kernel_wrapper_functional_proxy_570['DGAMMA'];  triton_kernel_wrapper_functional_proxy_570 = None
        empty_738: "bf16[512, 64, 12544]" = torch.ops.aten.empty.memory_format([512, 64, 12544], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_209: "bf16[512, 64, 12544]" = torch.ops.aten.permute.default(empty_738, [0, 1, 2]);  empty_738 = None
        triton_kernel_wrapper_functional_proxy_571 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 646, constant_args_idx = 882, grid = [(64, 6272, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3985, 'DY': view_3982, 'INVSTD': rsqrt, 'GAMMA': primals_4, 'DBETA': getitem_1305, 'DGAMMA': getitem_1306, 'DX': permute_209, 'M': 6422528, 'HW': 12544, 'stride_n': 802816, 'stride_c': 12544, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3985 = view_3982 = rsqrt = primals_4 = permute_209 = None
        getitem_1307: "bf16[512, 64, 12544]" = triton_kernel_wrapper_functional_proxy_571['DX'];  triton_kernel_wrapper_functional_proxy_571 = None
        convert_element_type_default_1: "f32[64]" = torch.ops.prims.convert_element_type.default(getitem_1306, torch.float32);  getitem_1306 = None
        convert_element_type_default: "f32[64]" = torch.ops.prims.convert_element_type.default(getitem_1305, torch.float32);  getitem_1305 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT5/layers.py:100 in forward, code: return _QuanConv2d.apply(x, self.weight,
        empty_739: "bf16[301056, 256]" = torch.ops.aten.empty.memory_format([301056, 256], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_572 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 647, constant_args_idx = 883, grid = [(301056, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem, 'S_ptr': getitem_1, 'M_ptr': getitem_2, 'Y_ptr': empty_739, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem = getitem_1 = getitem_2 = empty_739 = None
        getitem_1308: "bf16[301056, 256]" = triton_kernel_wrapper_functional_proxy_572['Y_ptr'];  triton_kernel_wrapper_functional_proxy_572 = None
        view_4002: "bf16[512, 64, 112, 112]" = torch.ops.aten.view.default(getitem_1307, [512, 64, 112, 112]);  getitem_1307 = None
        empty_740: "bf16[1]" = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_105: "bf16[64, 3, 7, 7]" = torch.ops.aten.expand.default(empty_740, [64, 3, 7, 7]);  empty_740 = None
        view_4003: "bf16[512, 588, 256]" = torch.ops.aten.view.default(getitem_1308, [512, 588, 256]);  getitem_1308 = None
        view_4004: "bf16[512, 150528]" = torch.ops.aten.view.default(view_4003, [512, -1]);  view_4003 = None
        view_4005: "bf16[512, 3, 224, 224]" = torch.ops.aten.view.default(view_4004, [512, 3, 224, 224]);  view_4004 = None
        convolution_backward_104 = torch.ops.aten.convolution_backward.default(view_4002, view_4005, expand_105, None, [2, 2], [3, 3], [1, 1], False, [0], 1, [False, True, False]);  view_4002 = view_4005 = expand_105 = None
        getitem_1310: "bf16[64, 3, 7, 7]" = convolution_backward_104[1];  convolution_backward_104 = None
        convert_element_type_327: "f32[64, 3, 7, 7]" = torch.ops.prims.convert_element_type.default(getitem_1310, torch.float32);  getitem_1310 = None
        return (convert_element_type_327, None, None, convert_element_type_default_1, convert_element_type_default, None, None, convert_element_type_322, None, convert_element_type_default_3, convert_element_type_default_2, None, None, convert_element_type_317, None, convert_element_type_default_5, convert_element_type_default_4, None, None, convert_element_type_312, None, convert_element_type_default_7, convert_element_type_default_6, None, None, convert_element_type_307, None, convert_element_type_default_9, convert_element_type_default_8, None, None, convert_element_type_302, None, convert_element_type_default_11, convert_element_type_default_10, None, None, convert_element_type_297, None, convert_element_type_default_13, convert_element_type_default_12, None, None, convert_element_type_292, None, convert_element_type_default_15, convert_element_type_default_14, None, None, convert_element_type_287, None, convert_element_type_default_17, convert_element_type_default_16, None, None, convert_element_type_282, None, convert_element_type_default_19, convert_element_type_default_18, None, None, convert_element_type_277, None, convert_element_type_default_21, convert_element_type_default_20, None, None, convert_element_type_272, None, convert_element_type_default_23, convert_element_type_default_22, None, None, convert_element_type_267, None, convert_element_type_default_25, convert_element_type_default_24, None, None, convert_element_type_262, None, convert_element_type_default_27, convert_element_type_default_26, None, None, convert_element_type_257, None, convert_element_type_default_29, convert_element_type_default_28, None, None, convert_element_type_252, None, convert_element_type_default_31, convert_element_type_default_30, None, None, convert_element_type_247, None, convert_element_type_default_33, convert_element_type_default_32, None, None, convert_element_type_242, None, convert_element_type_default_35, convert_element_type_default_34, None, None, convert_element_type_237, None, convert_element_type_default_37, convert_element_type_default_36, None, None, convert_element_type_232, None, convert_element_type_default_39, convert_element_type_default_38, None, None, convert_element_type_227, None, convert_element_type_default_41, convert_element_type_default_40, None, None, convert_element_type_222, None, convert_element_type_default_43, convert_element_type_default_42, None, None, convert_element_type_217, None, convert_element_type_default_45, convert_element_type_default_44, None, None, convert_element_type_212, None, convert_element_type_default_47, convert_element_type_default_46, None, None, convert_element_type_207, None, convert_element_type_default_49, convert_element_type_default_48, None, None, convert_element_type_202, None, convert_element_type_default_51, convert_element_type_default_50, None, None, convert_element_type_197, None, convert_element_type_default_53, convert_element_type_default_52, None, None, convert_element_type_192, None, convert_element_type_default_55, convert_element_type_default_54, None, None, convert_element_type_187, None, convert_element_type_default_57, convert_element_type_default_56, None, None, convert_element_type_182, None, convert_element_type_default_59, convert_element_type_default_58, None, None, convert_element_type_177, None, convert_element_type_default_61, convert_element_type_default_60, None, None, convert_element_type_172, None, convert_element_type_default_63, convert_element_type_default_62, None, None, convert_element_type_167, None, convert_element_type_default_65, convert_element_type_default_64, None, None, convert_element_type_162, None, convert_element_type_default_67, convert_element_type_default_66, None, None, convert_element_type_157, None, convert_element_type_default_69, convert_element_type_default_68, None, None, convert_element_type_152, None, convert_element_type_default_71, convert_element_type_default_70, None, None, convert_element_type_147, None, convert_element_type_default_73, convert_element_type_default_72, None, None, convert_element_type_142, None, convert_element_type_default_75, convert_element_type_default_74, None, None, convert_element_type_137, None, convert_element_type_default_77, convert_element_type_default_76, None, None, convert_element_type_132, None, convert_element_type_default_79, convert_element_type_default_78, None, None, convert_element_type_127, None, convert_element_type_default_81, convert_element_type_default_80, None, None, convert_element_type_122, None, convert_element_type_default_83, convert_element_type_default_82, None, None, convert_element_type_117, None, convert_element_type_default_85, convert_element_type_default_84, None, None, convert_element_type_112, None, convert_element_type_default_87, convert_element_type_default_86, None, None, convert_element_type_107, None, convert_element_type_default_89, convert_element_type_default_88, None, None, convert_element_type_102, None, convert_element_type_default_91, convert_element_type_default_90, None, None, convert_element_type_97, None, convert_element_type_default_93, convert_element_type_default_92, None, None, convert_element_type_92, None, convert_element_type_default_95, convert_element_type_default_94, None, None, convert_element_type_87, None, convert_element_type_default_97, convert_element_type_default_96, None, None, convert_element_type_82, None, convert_element_type_default_99, convert_element_type_default_98, None, None, convert_element_type_77, None, convert_element_type_default_101, convert_element_type_default_100, None, None, convert_element_type_72, None, convert_element_type_default_103, convert_element_type_default_102, None, None, convert_element_type_67, None, convert_element_type_default_105, convert_element_type_default_104, None, None, convert_element_type_62)
        