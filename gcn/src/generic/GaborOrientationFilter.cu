#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/GaborOrientationFilter.cu"
#else

int cugcn_(GOF_Producing)(
    THCTensor *weight,
    THCTensor *gaborFilterBank,
    THCTensor *output)
{
    THCUNN_assertSameGPU(state, 3, weight, gaborFilterBank, output);
    THArgCheck(weight->nDimension == 5, 1, "only supports a batch of GOFs.");
    const short nOutputPlane = weight->size[0];
    const uint16 nInputPlane = weight->size[1];
    const uint8 nChannel = weight->size[2];
    const uint8 kH = weight->size[3];
    const uint8 kW = weight->size[4];

    THCTensor_(resize4d)(state, output, nOutputPlane * nChannel, nInputPlane * nChannel, kH, kW);

    real *weightData = THCTensor_(data)(state, weight);
    real *gaborFilterBankData = THCTensor_(data)(state, gaborFilterBank);
    real *outputData = THCTensor_(data)(state, output);

    const uint16 nEntry = nChannel * kH * kW;
    const uint32 count = nOutputPlane * nInputPlane * nEntry;

    kernel_(GaborProducing)(
        THCState_getCurrentStream(state),
        count, 
        weightData, 
        gaborFilterBankData, 
        nInputPlane, 
        nOutputPlane, 
        nChannel, 
        nEntry, 
        outputData);
    THCudaCheck(cudaGetLastError());

    return 1;
}

int cugcn_(GOF_BPAlign)(
    THCTensor *weight,
    THCTensor *gaborFilterBank,
    THCTensor *gradWeight)
{
    THCUNN_assertSameGPU(state, 3, weight, gaborFilterBank, gradWeight);
    const uint8 nChannel = gaborFilterBank->size[0];
    const uint16 kH = gradWeight->size[2];;
    const uint16 kW = gradWeight->size[3];;
    const uint16 nOutputPlane = gradWeight->size[0] / nChannel;
    const uint16 nInputPlane = gradWeight->size[1] / nChannel;

    THCTensor_(resize5d)(state, weight, nOutputPlane, nInputPlane, nChannel, kH, kW);

    real *weightData = THCTensor_(data)(state, weight);
    real *gaborFilterBankData = THCTensor_(data)(state, gaborFilterBank);
    real *gradWeightData = THCTensor_(data)(state, gradWeight);

    const uint16 nEntry = nChannel * kH * kW;
    const uint32 count = nOutputPlane * nInputPlane * nEntry;

    kernel_(BPAlign)(
        THCState_getCurrentStream(state),
        count, 
        gradWeightData, 
        nInputPlane, 
        nOutputPlane, 
        nChannel, 
        kH,
        kW, 
        weightData,
        gaborFilterBankData);
    THCudaCheck(cudaGetLastError());

    return 1;
}

#endif