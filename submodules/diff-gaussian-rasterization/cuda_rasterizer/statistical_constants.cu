// statistical_constants.cu
#define STATISTICAL_CONSTANTS_IMPL
#include "statistical_constants.cuh"

__constant__ float BASE_SAMPLES_MAX[MAX_STD_SAMPLES * 3];
__constant__ int   NUM_STD_SAMPLES;
