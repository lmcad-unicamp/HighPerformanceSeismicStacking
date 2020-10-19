#include "cuda/include/semblance/kernel/zocrs/common.cuh"

__global__
void selectTracesForZeroOffsetCommonReflectionSurface(
    const float* midpointArray,
    unsigned int traceCount,
    unsigned char* usedTraceMaskArray,
    float m0,
    float apm
) {
    unsigned int traceIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (traceIndex < traceCount) {
        usedTraceMaskArray[traceIndex] = fabs(m0 - midpointArray[traceIndex]) <= apm;
    }
}
