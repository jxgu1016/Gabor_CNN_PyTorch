#!/bin/bash
HOME=$(pwd)
echo "Compiling cuda kernels..."
cd $HOME/gcn/src
rm libgcn_kernel.cu.o
nvcc -c -o libgcn_kernel.cu.o libgcn_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_35
echo "Installing extension..."
cd $HOME
python setup.py clean && python setup.py install
