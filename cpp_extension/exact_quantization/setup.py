from setuptools import setup, find_packages
from torch.utils import cpp_extension

setup(name='division',
      ext_modules=[
          cpp_extension.CUDAExtension(
              'calc_precision',
              ['calc_precision.cc']
          ),
          cpp_extension.CUDAExtension(
              'backward_func',
              ['backward_func.cc', 'backward_func_cuda_kernel.cu']
          ),
          cpp_extension.CUDAExtension(
              'quantization',
              ['quantization.cc', 'quantization_cuda_kernel.cu'],
              # extra_compile_args={'nvcc': ['--expt-extended-lambda']}
          ),
          # cpp_extension.CUDAExtension(
          #     'exact.cpp_extension.spmm',
          #     ['exact/cpp_extension/spmm.cc', 'exact/cpp_extension/spmm_cuda.cu']
          # ),
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      packages=find_packages()
)