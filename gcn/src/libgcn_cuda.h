typedef unsigned long uint64;
typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;

// THDoubleTensor -> THCudaDoubleTensor
// THFloatTensor -> THCudaTensor
int cugcn_Double_GOF_Producing(
    THCudaDoubleTensor *weight,
    THCudaDoubleTensor *gaborFilterBank,
    THCudaDoubleTensor *output);
int cugcn_Float_GOF_Producing(
    THCudaTensor *weight,
    THCudaTensor *gaborFilterBank,
    THCudaTensor *output);

int cugcn_Double_GOF_BPAlign(
    THCudaDoubleTensor *weight,
    THCudaDoubleTensor *gaborFilterBank,
    THCudaDoubleTensor *gradWeight);
int cugcn_Float_GOF_BPAlign(
    THCudaTensor *weight,
    THCudaTensor *gaborFilterBank,
    THCudaTensor *gradWeight);

