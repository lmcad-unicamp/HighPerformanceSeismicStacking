#pragma once

#include <cuda.h>

__global__
void buildParameterArrayForCommonMidPoint(
    float* parameterArray,
    float minVelocity,
    float increment,
    unsigned int totalParameterCount
);

__global__
void computeSemblancesForCommonMidPoint(
    const float *samples,
    const float *halfoffset,
    unsigned int traceCount,
    unsigned int samplesPerTrace,
    float dtInSeconds,
    int tauIndexDisplacement,
    int windowSize,
    /* Parameter arrays */
    const float *parameterArray,
    unsigned int totalParameterCount,
    /* Output arrays */
    float *semblanceArray,
    float *stackArray
);

__global__
void selectBestSemblancesForCommonMidPoint(
    const float *semblanceArray,
    const float *stackArray,
    const float *parameterArray,
    unsigned int totalParameterCount,
    unsigned int samplesPerTrace,
    float *resultArray
);

__global__
void selectTracesForCommonMidPoint(
    const float* midpointArray,
    unsigned int traceCount,
    unsigned char* usedTraceMaskArray,
    float m0
);
