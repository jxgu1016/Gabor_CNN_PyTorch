#pragma once
#include <torch/extension.h>


at::Tensor GOF_forward_cpu(const at::Tensor& weight, 
                           const at::Tensor& gaborFilterBank);

at::Tensor GOF_backward_cpu(const at::Tensor& grad_output,
                            const at::Tensor& gaborFilterBank);
