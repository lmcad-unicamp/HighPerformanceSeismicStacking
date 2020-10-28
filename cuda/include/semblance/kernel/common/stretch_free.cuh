#pragma once

#include <cuda.h>

__global__
void selectBestSemblances(
    const float *semblanceArray,
    const float *stackArray,
    const float *nArray,
    unsigned int totalNCount,
    unsigned int samplesPerTrace,
    float *resultArray
);