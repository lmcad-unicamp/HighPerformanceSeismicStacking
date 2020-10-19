#include "cuda/include/semblance/kernel/cmp/common.cuh"

__global__
void selectTracesForCommonMidPoint(
    const float* midpointArray,
    unsigned int traceCount,
    unsigned char* usedTraceMaskArray,
    float m0
) {
    unsigned int traceIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (traceIndex < traceCount) {
        usedTraceMaskArray[traceIndex] = (m0 == midpointArray[traceIndex]);
    }
}
