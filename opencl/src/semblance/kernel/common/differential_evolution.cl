#include "common/include/gpu/interface.h"

#define PRIME 197063L
#define UINT_MAX 4294967295

inline float random_next(__global unsigned int* state) {
    // XORWOW - CUDA's default pseudorandom number generator.
    // https://docs.nvidia.com/cuda/curand/device-api-overview.html
    // https://en.wikipedia.org/wiki/Xorshift#xorwow

    unsigned int t = state[4];
    unsigned int s = state[0];
    state[4] = state[3];
    state[3] = state[2];
    state[2] = state[1];
    state[1] = s;
    t ^= t >> 2;
    t ^= t << 1;
    t ^= s ^ (s << 4);
    state[0] = t;
    state[5] += 362437;

    return (float) (t + state[5]) / (float) UINT_MAX;
}

__kernel
void startPopulations(
    __global __write_only float *x,
    __global __read_only float *min,
    __global __read_only float *max,
    __global unsigned int *st,
    unsigned int individualsPerPopulation,
    unsigned int samplesPerTrace,
    unsigned int numberOfParameters
) {
    unsigned int threadIndex = get_group_id(0) * get_local_size(0) + get_local_id(0);
    unsigned int sampleIndex = threadIndex / individualsPerPopulation;
    unsigned int individualIndex = threadIndex % individualsPerPopulation;
    unsigned int seedIndex = 6 * threadIndex;

    if (sampleIndex < samplesPerTrace) {
        unsigned int offset = (sampleIndex * individualsPerPopulation + individualIndex) * numberOfParameters;
        for (unsigned int parameterIndex = 0; parameterIndex < numberOfParameters; parameterIndex++) {
            float ratio = random_next(&st[seedIndex]);
            x[offset + parameterIndex] = min[parameterIndex] + ratio * (max[parameterIndex] - min[parameterIndex]);
        }
    }
}

__kernel
void mutatePopulations(
    __global __write_only float *v,
    __global __read_only float *x,
    __global __read_only float *minArray,
    __global __read_only float *maxArray,
    __global unsigned int *st,
    unsigned int individualsPerPopulation,
    unsigned int samplesPerTrace,
    unsigned int numberOfParameters
) {
    unsigned int threadIndex = get_group_id(0) * get_local_size(0) + get_local_id(0);
    unsigned int sampleIndex = threadIndex / individualsPerPopulation;
    unsigned int individualIndex = threadIndex % individualsPerPopulation;
    unsigned int seedIndex = 6 * threadIndex;

    if (sampleIndex < samplesPerTrace) {
        unsigned int popIndex = sampleIndex * individualsPerPopulation * numberOfParameters;
        unsigned int offset = popIndex + individualIndex * numberOfParameters;

        float p1, p2, p3;
        unsigned int r1, r2, r3;
        for (unsigned int parameterIndex = 0; parameterIndex < numberOfParameters; parameterIndex++) {
            p1 = random_next(&st[seedIndex]);
            p2 = random_next(&st[seedIndex]);
            p3 = random_next(&st[seedIndex]);

            r1 = popIndex +
                ((unsigned int) (p1 * (float)(individualsPerPopulation - 1))) * numberOfParameters +
                parameterIndex;

            r2 = popIndex +
                ((unsigned int) (p2 * (float)(individualsPerPopulation - 1))) * numberOfParameters +
                parameterIndex;

            r3 = popIndex +
                ((unsigned int) (p3 * (float)(individualsPerPopulation - 1))) * numberOfParameters +
                parameterIndex;

            float newIndividual = x[r1] + F_FAC * (x[r2] - x[r3]);
            newIndividual = min(newIndividual, maxArray[parameterIndex]);
            newIndividual = max(newIndividual, minArray[parameterIndex]);

            v[offset + parameterIndex] = newIndividual;
        }
    }
}

__kernel
void crossoverPopulations(
    __global __write_only float *u,
    __global __read_only float *x,
    __global __read_only float *v,
    __global unsigned int *st,
    unsigned int individualsPerPopulation,
    unsigned int samplesPerTrace,
    unsigned int numberOfParameters
) {
    unsigned int threadIndex = get_group_id(0) * get_local_size(0) + get_local_id(0);
    unsigned int sampleIndex = threadIndex / individualsPerPopulation;
    unsigned int individualIndex = threadIndex % individualsPerPopulation;
    unsigned int seedIndex = 6 * threadIndex;

    if (sampleIndex < samplesPerTrace) {
        unsigned int l = (unsigned int)(random_next(&st[seedIndex]) * (float)(numberOfParameters - 1));
        unsigned int offset = (sampleIndex * individualsPerPopulation + individualIndex) * numberOfParameters;

        for (unsigned int parameterIndex = 0; parameterIndex < numberOfParameters; parameterIndex++) {
            float r = random_next(&st[seedIndex]);
            unsigned int featureIndex = offset + parameterIndex;

            if (r > CR && l != parameterIndex) {
                u[featureIndex] = x[featureIndex];
            }
            else {
                u[featureIndex] = v[featureIndex];
            }
        }
    }
}

__kernel
void nextGeneration(
    __global float *x,
    __global float *fx,
    __global __read_only float *u,
    __global __read_only float *fu,
    unsigned int individualsPerPopulation,
    unsigned int samplesPerTrace,
    unsigned int numberOfParameters,
    unsigned int numberOfCommonResults
) {
    unsigned int threadIndex = get_group_id(0) * get_local_size(0) + get_local_id(0);
    unsigned int sampleIndex = threadIndex / individualsPerPopulation;
    unsigned int individualIndex = threadIndex % individualsPerPopulation;

    if (sampleIndex < samplesPerTrace) {
        unsigned int popIndex = (sampleIndex * individualsPerPopulation + individualIndex) * numberOfParameters;
        unsigned int fitnessIndex = (sampleIndex * individualsPerPopulation + individualIndex) * numberOfCommonResults;

        if (fu[fitnessIndex] > fx[fitnessIndex]) {
            for (unsigned int parameterIndex = 0; parameterIndex < numberOfParameters; parameterIndex++) {
                unsigned int featureIndex = popIndex + parameterIndex;
                x[featureIndex] = u[featureIndex];
            }

            for (unsigned int resultIndex = 0; resultIndex < numberOfCommonResults; resultIndex++) {
                unsigned int perfIndex = fitnessIndex + resultIndex;
                fx[perfIndex] = fu[perfIndex];
            }
        }
    }
}
