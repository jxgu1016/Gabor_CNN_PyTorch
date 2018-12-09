#pragma once

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

// Interface for Python
at::Tensor GOF_forward(const at::Tensor& weight, 
                       const at::Tensor& gaborFilterBank) {
  if (weight.type().is_cuda()) {
#ifdef WITH_CUDA
    return GOF_forward_cuda(weight, gaborFilterBank);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return GOF_forward_cpu(weight, gaborFilterBank);
}

at::Tensor GOF_backward(const at::Tensor& grad_output,
                        const at::Tensor& gaborFilterBank) {
  if (grad_output.type().is_cuda()) {
#ifdef WITH_CUDA
    return GOF_backward_cuda(grad_output, gaborFilterBank);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return GOF_backward_cpu(grad_output, gaborFilterBank);
}

