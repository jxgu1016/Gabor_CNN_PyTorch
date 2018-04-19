typedef unsigned long uint64;
typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;

int gcn_Float_GOF_Producing(
    THFloatTensor *weight,
    THFloatTensor *gaborFilterBank,
    THFloatTensor *output);
int gcn_Double_GOF_Producing(
    THDoubleTensor *weight,
    THDoubleTensor *gaborFilterBank,
    THDoubleTensor *output);

int gcn_Float_GOF_BPAlign(
    THFloatTensor *weight,
    THFloatTensor *gaborFilterBank,
    THFloatTensor *gradWeight);
int gcn_Double_GOF_BPAlign(
    THDoubleTensor *weight,
    THDoubleTensor *gaborFilterBank,
    THDoubleTensor *gradWeight);