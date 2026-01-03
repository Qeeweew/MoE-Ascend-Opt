import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# ARM-only build (Ascend NPU + ARM CPU)
if os.environ.get("CXX") is None:
    os.environ["CXX"] = "g++"

ASCEND_HOME = os.environ.get("ASCEND_HOME", "/home/xwj/Ascend/ascend-toolkit/latest")

setup(
    name="nanovllm_ext",
    ext_modules=[
        CppExtension(
            "nanovllm_ext",
            [
                "q8_gemm.cpp",
                "moe_infer.cpp",
                "pybind.cpp",
            ],
            define_macros=[
                ("WITH_NPU", None),
            ],
            extra_compile_args=[
                "-O3",
                "-ffast-math",
                "-Wall",
                "-fopenmp",
                "-march=armv8.2-a+dotprod+fp16",
                "-std=c++17",
            ],
            include_dirs=[
                f"{ASCEND_HOME}/include",
            ],
            library_dirs=[
                f"{ASCEND_HOME}/lib64",
            ],
            extra_link_args=[
                "-fopenmp",
                "-lascendcl",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
