import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch_npu.utils.cpp_extension import NpuExtension

setup(
    name="nanovllm_ext",
    ext_modules=[
        NpuExtension(
            name="nanovllm_ext",
            sources=[
                "q8_gemm.cpp",
                "moe_infer.cpp",
                "nanovllm_ops.cpp",
            ],
            define_macros=[("WITH_NPU", None)],
            extra_compile_args=[
                "-O3",
                "-ffast-math",
                "-Wall",
                "-fopenmp",
                "-march=armv8.2-a+dotprod+fp16",
                "-std=c++17",
            ],
            extra_link_args=[
                "-fopenmp",
                "-lnuma",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
