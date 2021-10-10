from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension

setup(name='actnn',
      ext_modules=[
          cpp_extension.CUDAExtension(
              'calc_precision',
              ['calc_precision.cc']
          ),
          cpp_extension.CUDAExtension(
              'minimax',
              ['minimax.cc', 'minimax_cuda_kernel.cu']
          ),
          cpp_extension.CUDAExtension(
              'backward_func',
              ['backward_func.cc']
          ),
          cpp_extension.CUDAExtension(
              'quantization',
              ['quantization.cc', 'quantization_cuda_kernel.cu']
          ),
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      packages=find_packages()
)
