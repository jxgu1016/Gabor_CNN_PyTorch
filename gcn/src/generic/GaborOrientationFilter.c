#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/GaborOrientationFilter.c"
#else

int gcn_(GOF_Producing)(
    THTensor *weight,
    THTensor *gaborFilterBank,
    THTensor *output)
{
    THArgCheck(weight->nDimension == 5, 1, "only supports a batch of GOFs.");
    const uint16 nOutputPlane = weight->size[0];
    const uint16 nInputPlane = weight->size[1];
    const uint8 nChannel = weight->size[2];
    const uint8 kH = weight->size[3];
    const uint8 kW = weight->size[4];

    THTensor_(resize4d)(output, nOutputPlane * nChannel, nInputPlane * nChannel, kH, kW);

    real *weightData = THTensor_(data)(weight);
    real *gaborFilterBankData = THTensor_(data)(gaborFilterBank);
    real *outputData = THTensor_(data)(output);
	
    uint32 i, j, l, k;
  
    #pragma omp parallel for private(i, j, l, k)
    for (i = 0; i < nOutputPlane; i++) {
        for (j = 0; j < nInputPlane; j++) {
            
            for (l = 0; l < nChannel * kH * kW; l++) {
                real val = *(weightData + i * (nInputPlane * nChannel * kH * kW)
                                        + j * (nChannel * kH * kW)
                                        + l);
										              
                for (k = 0; k < nChannel; k++) {
                    real gabortmp = *(gaborFilterBankData + k * (kW * kH) + l % (kW * kH));

                        real *target = outputData + i * (nChannel * nInputPlane * nChannel * kH * kW)
                                                + k * (nInputPlane * nChannel * kH * kW)
                                                + j * (nChannel * kH * kW)
                                                + l;
                        *target = val * gabortmp;
                }
            }
        }
    }

    return 1;
}

int gcn_(GOF_BPAlign)(
    THTensor *weight,
    THTensor *gaborFilterBank,
    THTensor *gradWeight)
{
    const uint8 nChannel = gaborFilterBank->size[0];
    const uint16 nOutputPlane = gradWeight->size[0] / nChannel;
    const uint16 nInputPlane = gradWeight->size[1] / nChannel;
    const uint8 kH = gradWeight->size[2];
    const uint8 kW = gradWeight->size[3];

    THTensor_(resize5d)(weight, nOutputPlane, nInputPlane, nChannel, kH, kW);

    real *weightData = THTensor_(data)(weight);
    real *gaborFilterBankData = THTensor_(data)(gaborFilterBank);
    real *gradWeightData = THTensor_(data)(gradWeight);

    const uint32 nEntry = nChannel * kH * kW;
    uint32 i, j, l, k;

    #pragma omp parallel for private(i, j, l, k)
    for (i = 0; i < nOutputPlane; i++) {
        for (j = 0; j < nInputPlane; j++) {

            for (l = 0; l < nEntry; l++) {
                real *val = weightData + i * (nInputPlane * nEntry)
                                       + j * (nEntry)
                                       + l;
                *val = 0; ////////////////HERE'S THE PROBLEM!!!!!!!!!!
                for (k = 0; k < nChannel; k++) {
                    real gabortmp = *(gaborFilterBankData + k * (kW * kH) + l % (kW * kH));
                    real *target = gradWeightData + i * (nChannel * nInputPlane * nEntry)
                                                  + k * (nInputPlane * nEntry)
                                                  + j * (nEntry)
                                                  + l;
                    *val = *val + *target * gabortmp;
					/**val = *val + *target;*/
                }
            }
        }
    }
    return 1;
}
#endif
