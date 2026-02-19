from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Collect all CUDA/C++ sources
sources = [
    # Dispatch + launchers
    "quant_pack_dispatch.cpp",
    "quant_pack_launchers.cu",


    # ---------------------------
    # 1-bit kernels (4 architectures)
    # ---------------------------
    "quant_pack_1bit_sm80.cu",
    #"quant_pack_1bit_sm89.cu",
    "quant_pack_1bit_sm90.cu",
    "quant_pack_1bit_sm90a.cu",

    # ---------------------------
    # 2-bit kernels
    # ---------------------------
    "quant_pack_2bit_sm80.cu",
    "quant_pack_2bit_sm89.cu",
    "quant_pack_2bit_sm90.cu",
    "quant_pack_2bit_sm90a.cu",

    # ---------------------------
    # 4-bit kernels
    # ---------------------------
    "quant_pack_4bit_sm80.cu",
    "quant_pack_4bit_sm89.cu",
    "quant_pack_4bit_sm90.cu",
    "quant_pack_4bit_sm90a.cu",

    # ---------------------------
    # 8-bit kernels
    # ---------------------------
    "quant_pack_8bit_sm80.cu",
    "quant_pack_8bit_sm89.cu",
    "quant_pack_8bit_sm90.cu",
    "quant_pack_8bit_sm90a.cu",
]

# Ensure NVCC sees correct architectures
extra_cuda_flags = [
    "-O3",
    "--use_fast_math",

    # Allow half/bfloat16 intrinsics
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",

    # Modern GPUs
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_90,code=sm_90",
    "-gencode=arch=compute_90a,code=sm_90a",

    # Remove warnings
    "-Xcudafe=--diag_suppress=177"
]

setup(
    name="quantizer",
    ext_modules=[
        CUDAExtension(
            name="quantizer",
            sources=sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": extra_cuda_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
