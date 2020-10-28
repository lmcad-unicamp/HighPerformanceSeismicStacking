#pragma once

#include <cuda.h>

__global__
void computeSemblancesForOffsetContinuationTrajectory(
    const float *samples,
    const float *midpoint,
    const float *halfoffset,
    unsigned int traceCount,
    unsigned int samplesPerTrace,
    float apm,
    float m0,
    float h0,
    float dtInSeconds,
    int tauIndexDisplacement,
    int windowSize,
    /* Parameter arrays */
    const float *parameterArray,
    const float *nArray,
    unsigned int totalNCount,
    /* Output arrays */
    float* notUsedCountArray,
    float *semblanceArray,
    float *stackArray
);

__global__
void selectTracesForOffsetContinuationTrajectory(
    const float *midpointArray,
    const float *halfoffsetArray,
    const float *parameterArray,
    unsigned int traceCount,
    unsigned int samplesPerTrace,
    float dtInSeconds,
    float apm,
    float m0,
    float h0,
    unsigned char* usedTraceMaskArray
);
