#pragma once

#include "common/include/gpu/interface.h"
#include <cuda.h>
#include <curand_kernel.h>

__global__
void kernelDifferentialEvolution(
    const float *samples,
    const float *midpoint,
    const float *halfoffset,
    unsigned int traceCount,
    gpu_gather_data_t gatherData,
    gpu_reference_point_t referencePoint,
    gpu_traveltime_data_t traveltime,
    const float *parameterArray,
    float *resultArray,
    float* notUsedCountArray
);

__global__
void filterOutTracesForOffsetContinuationTrajectoryAndDifferentialEvolution(
    const float *midpointArray,
    const float *halfoffsetArray,
    unsigned char* mustUseTraceArray,
    unsigned int traceCount,
    gpu_gather_data_t gatherData,
    gpu_reference_point_t referencePoint,
    gpu_traveltime_data_t traveltime,
    const float *parameterArray,
    unsigned int parameterCount,
    unsigned int arrayStep
);

__global__
void kernelSetupRandomSeedArray(
    curandState *state,
    unsigned int seed
);

__global__
void kernelStartPopulations(
    float* x,
    const float* min,
    const float* max,
    curandState *st,
    unsigned int numberOfParameters
);

__global__
void kernelmutateAllPopulations(
    float* v,
    const float* x,
    const float* min,
    const float* max,
    curandState *st,
    unsigned int numberOfParameters
);

__global__
void kernelCrossoverPopulationIndividuals(
    float* u,
    const float* x,
    const float* v,
    curandState *st,
    unsigned int numberOfParameters
);

__global__
void kernelAdvanceGeneration(
    float* x,
    float* fx,
    const float* u,
    const float* fu,
    unsigned int numberOfParameters,
    unsigned int numberOfCommonResults
);

__global__
void kernelSelectBestIndividuals(
    const float* x,
    const float* fx,
    float* resultArray,
    unsigned int numberOfParameters,
    unsigned int numberOfCommonResults,
    unsigned int numberOfSamples
);
