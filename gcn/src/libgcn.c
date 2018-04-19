#include <TH/TH.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "libgcn.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch., Real, Tensor)
#define gcn_(NAME) TH_CONCAT_4(gcn_, Real, _, NAME)

#include "generic/GaborOrientationFilter.c"
#include "THGenerateFloatTypes.h"

//Strange Error here -> looks like THGenerateFloatTypes include double-types
// #include "generic/GaborOrientationFilter.c" 
// #include "THGenerateDoubleType.h"