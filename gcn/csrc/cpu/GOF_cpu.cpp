#include "cpu/vision.h"


template <typename T>
void GOFForward_cpu_kernel(
  const T* weight_data,
  const T* gaborFilterBank_data,
  const int nOutputPlane,
  const int nInputPlane,
  const int nChannel,
  const int kH,
  const int kW,
  T* output_data) {
  for (int i = 0; i < nOutputPlane; i++) {
    for (int j = 0; j < nInputPlane; j++) {
      for (int l = 0; l < nChannel * kH * kW; l++) {
        T val = *(weight_data + i * (nInputPlane * nChannel * kH * kW)
                              + j * (nChannel * kH * kW)
                              + l);
        for (int k = 0; k < nChannel; k++) {
          T gabortmp = *(gaborFilterBank_data + k * (kW * kH) 
                                              + l % (kW * kH));
          T *target = output_data + i * (nChannel * nInputPlane * nChannel * kH * kW)
                                  + k * (nInputPlane * nChannel * kH * kW)
                                  + j * (nChannel * kH * kW)
                                  + l;
          *target = val * gabortmp;
        }
      }
    }
  }
}

template <typename T>
void GOFBackward_cpu_kernel(
  const T* grad_output_data,
  const T* gaborFilterBank_data,
  const int nOutputPlane,
  const int nInputPlane,
  const int nChannel,
  const int kH,
  const int kW,
  T* grad_weight_data) {
  const int nEntry = nChannel * kH * kW;

  for (int i = 0; i < nOutputPlane; i++) {
    for (int j = 0; j < nInputPlane; j++) {
      for (int l = 0; l < nEntry; l++) {
        T *val = grad_weight_data + i * (nInputPlane * nEntry)
                                  + j * (nEntry) + l;
        *val = 0;
        for (int k = 0; k < nChannel; k++) {
          T gabortmp = *(gaborFilterBank_data + k * (kW * kH)
                                              + l % (kW * kH));
          T target = *(grad_output_data + i * (nChannel * nInputPlane * nEntry)
                                       + k * (nInputPlane * nEntry)
                                       + j * (nEntry)
                                       + l);
          *val = *val + target * gabortmp;
        }
      }
    }
  }
}


at::Tensor GOF_forward_cpu(const at::Tensor& weight,
                           const at::Tensor& gaborFilterBank) {
  AT_ASSERTM(!weight.type().is_cuda(), "weight must be a CPU tensor");
  AT_ASSERTM(!gaborFilterBank.type().is_cuda(), "gaborFilterBank must be a CPU tensor");

  auto nOutputPlane = weight.size(0);
  auto nInputPlane = weight.size(1);
  auto nChannel = weight.size(2);
  auto kH = weight.size(3);
  auto kW = weight.size(4);

  auto output = at::empty({nOutputPlane * nChannel, nInputPlane * nChannel, kH, kW}, weight.options());

  if (output.numel() == 0) {
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES(weight.type(), "GOF_forward", [&] {
    GOFForward_cpu_kernel<scalar_t>(
         weight.data<scalar_t>(),
         gaborFilterBank.data<scalar_t>(),
         nOutputPlane,
         nInputPlane,
         nChannel,
         kH,
         kW,
         output.data<scalar_t>());
  });
  return output;
}

at::Tensor GOF_backward_cpu(const at::Tensor& grad_output,
                            const at::Tensor& gaborFilterBank) {
  AT_ASSERTM(!grad_output.type().is_cuda(), "grad_output must be a CPU tensor");
  AT_ASSERTM(!gaborFilterBank.type().is_cuda(), "gaborFilterBank must be a CPU tensor");

  auto nChannel = gaborFilterBank.size(0);
  auto nOutputPlane = grad_output.size(0) / nChannel;
  auto nInputPlane = grad_output.size(1) / nChannel;
  auto kH = grad_output.size(2);
  auto kW = grad_output.size(3);

  auto grad_weight = at::empty({nOutputPlane, nInputPlane, nChannel, kH, kW}, grad_output.options());

  if (grad_weight.numel() == 0) {
    return grad_weight;
  }

  AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "GOF_backward", [&] {
    GOFBackward_cpu_kernel<scalar_t>(
         grad_output.data<scalar_t>(),
         gaborFilterBank.data<scalar_t>(),
         nOutputPlane,
         nInputPlane,
         nChannel,
         kH,
         kW,
         grad_weight.data<scalar_t>());
  });
  return grad_weight;
}