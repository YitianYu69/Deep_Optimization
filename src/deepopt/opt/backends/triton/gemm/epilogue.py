"""Bias addition and output cast for a native FP32 FP8 product."""

import torch
import triton
import triton.language as tl


@triton.jit
def _bias_output(input_ptr,bias_ptr,output_ptr,ELEMENTS:tl.constexpr,COLS:tl.constexpr,
                 STRIDE0:tl.constexpr,STRIDE1:tl.constexpr,BLOCK:tl.constexpr):
    index=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
    value=tl.load(input_ptr+(index//COLS)*STRIDE0+(index%COLS)*STRIDE1,index<ELEMENTS,other=0.0).to(tl.float32)
    bias=tl.load(bias_ptr+index%COLS,index<ELEMENTS,other=0.0).to(tl.float32)
    tl.store(output_ptr+index,(value+bias).to(output_ptr.dtype.element_ty),index<ELEMENTS)


def bias_output(product,bias,dtype):
    output=torch.empty(product.shape,device=product.device,dtype=dtype)
    _bias_output[(triton.cdiv(product.numel(),1024),)](product,bias,output,product.numel(),product.shape[1],*product.stride(),1024,
                                                       enable_fp_fusion=False)
    return output
