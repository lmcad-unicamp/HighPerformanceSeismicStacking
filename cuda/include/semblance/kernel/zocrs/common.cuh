#pragma once

#include <cuda.h>

__global__
void selectTracesForZeroOffsetCommonReflectionSurface(
    const float* midpointArray,
    unsigned int traceCount,
    unsigned char* usedTraceMaskArray,
    float m0,
    float apm
);
