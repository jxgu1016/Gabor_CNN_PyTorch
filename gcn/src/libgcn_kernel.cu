#include "libgcn_kernel.h"

#define FLT_MAX 3.402823466e+38F


template <typename Dtype>
__global__ void GaborProducingKernel(
    const uint32 nthreads, 
    const Dtype* weight_data,
    const Dtype* gaborFilterBank_data,
    const uint16 nInputPlane,
    const uint16 nOutputPlane,
    const uint8 nChannel,
    const uint16 nEntry,
    Dtype* output_data) 
{
    CUDA_KERNEL_LOOP(n, nthreads) {
        long l = n % nEntry;
        long j = (n / nEntry) % nInputPlane;
        long i = n / nEntry / nInputPlane;
        long k;
        Dtype val = *(weight_data + n);
        for (k = 0; k < nChannel; k++) {
            Dtype gabortmp=*(gaborFilterBank_data+k*(nEntry / nChannel)+l%(nEntry / nChannel));
            Dtype *target = output_data + i * (nChannel * nInputPlane * nEntry)
                                        + k * (nInputPlane * nEntry)
                                        + j * (nEntry)
                                        + l;
            *target = val*gabortmp;
        }
    }
}

template <typename Dtype>
__global__ void BPAlignKernel(
    const uint32 nthreads, 
    const Dtype* gradWeight_data,
    const uint16 nInputPlane,
    const uint16 nOutputPlane,
    const uint8 nChannel,
    const uint16 kH,
    const uint16 kW,
    Dtype* weight_data,
    const Dtype* gaborFilterBank_data) 
{
    const uint16 nEntry=nChannel*kH*kW; ///////////
    CUDA_KERNEL_LOOP(n, nthreads) {
        long l = n % nEntry;
        long j = (n / nEntry) % nInputPlane;
        long i = n / nEntry / nInputPlane;
        long k;
        Dtype *val = weight_data + n;
        *val = 0;
        for (k = 0; k < nChannel; k++) {
            Dtype gabortmp=*(gaborFilterBank_data+k*(kW*kH)+l%(kW*kH));
            Dtype target = *(gradWeight_data + i * (nChannel * nInputPlane * nEntry)
                                             + k * (nInputPlane * nEntry)
                                             + j * (nEntry)
                                             + l);
            
			*val = *val + target*gabortmp;
			//*val = *val + target;
        }
    }
}

#ifdef __cplusplus
extern "C" {
#endif

void kernel_Double_GaborProducing(
    cudaStream_t stream,
    const uint32 count, 
    const double* weight_data,
    const double* gaborFilterBank_data,
    const uint16 nInputPlane,
    const uint16 nOutputPlane,
    const uint8 nChannel,
    const uint16 nEntry,
    double* output_data)
{
    GaborProducingKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>
        (count, weight_data, gaborFilterBank_data, nInputPlane, nOutputPlane, nChannel, nEntry, output_data);
}

void kernel_Float_GaborProducing(
    cudaStream_t stream,
    const uint32 count, 
    const float* weight_data,
    const float* gaborFilterBank_data,
    const uint16 nInputPlane,
    const uint16 nOutputPlane,
    const uint8 nChannel,
    const uint16 nEntry,
    float* output_data)
{
    GaborProducingKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>
        (count, weight_data, gaborFilterBank_data, nInputPlane, nOutputPlane, nChannel, nEntry, output_data);
}

void kernel_Double_BPAlign(
    cudaStream_t stream,
    const uint32 count, 
    const double* gradWeight_data,
    const uint16 nInputPlane,
    const uint16 nOutputPlane,
    const uint8 nChannel,
    const uint16 kH,
    const uint16 kW,
    double* weight_data,
    const double* gaborFilterBank_data)
{
    BPAlignKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>
        (count, gradWeight_data, nInputPlane, nOutputPlane, nChannel, kH, kW, weight_data, gaborFilterBank_data);
}

void kernel_Float_BPAlign(
    cudaStream_t stream,
    const uint32 count, 
    const float* gradWeight_data,
    const uint16 nInputPlane,
    const uint16 nOutputPlane,
    const uint8 nChannel,
    const uint16 kH,
    const uint16 kW,
    float* weight_data,
    const float* gaborFilterBank_data)
{
    BPAlignKernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>
        (count, gradWeight_data, nInputPlane, nOutputPlane, nChannel, kH, kW, weight_data, gaborFilterBank_data);
}

#ifdef __cplusplus
}
#endif