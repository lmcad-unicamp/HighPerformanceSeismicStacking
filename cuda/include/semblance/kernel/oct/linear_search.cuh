#pragma once

#include <cuda.h>

__global__
void buildParameterArrayForOffsetContinuationTrajectory(
    float* parameterArray,
    float minVelocity,
    float incrementVelocity,
    unsigned int countVelocity,
    float minSlope,
    float incrementSlope,
    unsigned int countSlope,
    unsigned int totalParameterCount
);


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
    unsigned int totalParameterCount,
    /* Output arrays */
    float* notUsedCountArray,
    float *semblanceArray,
    float *stackArray
);

__global__
void selectBestSemblancesForOffsetContinuationTrajectory(
    const float *semblanceArray,
    const float *stackArray,
    const float *parameterArray,
    unsigned int totalParameterCount,
    unsigned int samplesPerTrace,
    float *resultArray
);

__global__
void selectTracesForOffsetContinuationTrajectory(
    const float *midpointArray,
    const float *halfoffsetArray,
    const float *parameterArray,
    unsigned int parameterCount,
    unsigned int traceCount,
    unsigned int samplesPerTrace,
    float dtInSeconds,
    float apm,
    float m0,
    float h0,
    unsigned char* usedTraceMaskArray
);