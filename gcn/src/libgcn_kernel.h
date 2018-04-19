typedef unsigned long uint64;
typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// Use 1024 threads per block, which requires cuda sm_2x or above
const int CUDA_NUM_THREADS = 1024;
 
// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
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
    double* output_data);
void kernel_Float_GaborProducing(
    cudaStream_t stream,
    const uint32 count, 
    const float* weight_data,
    const float* gaborFilterBank_data,
    const uint16 nInputPlane,
    const uint16 nOutputPlane,
    const uint8 nChannel,
    const uint16 nEntry,
    float* output_data);
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
    const double* gaborFilterBank_data);
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
    const float* gaborFilterBank_data);



#ifdef __cplusplus
}
#endif