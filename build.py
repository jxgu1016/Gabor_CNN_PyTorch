import os
import torch
from torch.utils.ffi import create_extension
import sys

this_file = os.path.dirname(__file__)

sources = ['gcn/src/libgcn.c']
headers = ['gcn/src/libgcn.h']
extra_objects = []
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['gcn/src/libgcn_cuda.c']
    headers += ['gcn/src/libgcn_cuda.h']
    extra_objects += ['gcn/src/libgcn_kernel.cu.o']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

extra_compile_args = None if sys.platform == 'darwin' else ['-fopenmp'] # MacOS does not support 'fopenmp'
ffi = create_extension(
    'gcn._ext.libgcn',
    package=True,
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    include_dirs=['gcn/src'],
    with_cuda=with_cuda,
    extra_objects=extra_objects,
    extra_compile_args=extra_compile_args,
)

if __name__ == '__main__':
    ffi.build()