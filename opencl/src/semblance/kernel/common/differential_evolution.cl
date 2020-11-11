#include "common/include/gpu/interface.h"
#include <limits.h>

#define PRIME 197063L

inline float random_next(__global unsigned int* seed) {
    // Lehmer random number generator
    // https://en.wikipedia.org/wiki/Lehmer_random_number_generator
    unsigned long currentSeed = (unsigned long) *seed;
    unsigned long product = currentSeed * PRIME;
    *seed = product % UINT_MAX;

    return  (float) (*seed) / (float) UINT_MAX;
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
    unsigned int seedIndex = threadIndex;

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
    unsigned int seedIndex = threadIndex;

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
                (unsigned int) (p2 * (float)(individualsPerPopulation - 1)) * numberOfParameters +
                parameterIndex;

            r2 = popIndex +
                (unsigned int) (p2 * (float)(individualsPerPopulation - 1)) * numberOfParameters +
                parameterIndex;

            r3 = popIndex +
                (unsigned int) (p2 * (float)(individualsPerPopulation - 1)) * numberOfParameters +
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
    unsigned int seedIndex = threadIndex;

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
