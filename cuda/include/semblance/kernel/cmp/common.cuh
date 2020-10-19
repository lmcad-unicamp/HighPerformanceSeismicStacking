#pragma once

#include <cuda.h>

__global__
void selectTracesForCommonMidPoint(
    const float* midpointArray,
    unsigned int traceCount,
    unsigned char* usedTraceMaskArray,
    float m0
);
