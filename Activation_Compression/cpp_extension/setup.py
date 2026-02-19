from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, include_paths
import torch
import glob
import os

import os
this_dir = os.path.dirname(os.path.abspath(__file__))

include_dirs = [
    os.path.join(this_dir, "include"),
    os.path.join(this_dir),                      # allow local includes
    *torch.utils.cpp_extension.include_paths(),
]

# Gather all CUDA and C++ sources automatically
kernel_sources = glob.glob("kernels/*.cu")
other_cuda = glob.glob("src/*.cu")
cpp_sources = glob.glob("src/*.cpp") + glob.glob("src/*.cc") + ["bind.cpp"]

print("Building with:")
print("  kernels =", kernel_sources)
print("  other cu =", other_cuda)
print("  cpp =", cpp_sources)

setup(
    name="cpp_extension",
    ext_modules=[
        CUDAExtension(
            name="cpp_extension",
            sources=cpp_sources + kernel_sources + other_cuda,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-use_fast_math",
                    "--extended-lambda",
                    "-Xfatbin=-compress-all",
                    "-lineinfo",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_89,code=sm_89",
                    "-gencode=arch=compute_90,code=sm_90",
                    "-gencode=arch=compute_90a,code=sm_90a",
                    "-Xcudafe=--diag_suppress=177"
                ]
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
