#include "common/include/gpu/interface.h"
#include "cuda/include/semblance/kernel/oct/linear_search.cuh"

__device__
enum gpu_error_code computeDisplacedMidpoint(float h, float h0, float t0, float m0, float c, float slope, float* mh) {
    float theta, theta_sq, gamma, gamma_sq, tn0_sq, tn0_quad, sqrt_arg;

    theta = t0 * slope;
    gamma = 2 * sqrt(h * h + h0 * h0);
    tn0_sq = t0 * t0 - c * h0 * h0;

    theta_sq = theta * theta;
    gamma_sq = gamma * gamma;
    tn0_quad = tn0_sq * tn0_sq;

    sqrt_arg = tn0_quad * tn0_quad + tn0_quad * theta_sq * gamma_sq + 16 * h * h * h0 * h0 * theta_sq * theta_sq;

    if (sqrt_arg < 0) {
        return NEGATIVE_SQUARED_ROOT;
    }

    sqrt_arg = theta_sq * gamma_sq + 2 * tn0_quad + 2 * sqrt(sqrt_arg);

    if (sqrt_arg == 0) {
        return DIVISION_BY_ZERO;
    }

    *mh = m0 + 2 * theta * (h * h - h0 * h0) / sqrt(sqrt_arg);

    return NO_ERROR;
}

__device__
enum gpu_error_code computeTime(float h, float h0, float t0, float m0, float m, float mh, float c, float slope, float* out) {
    float tn0_sq, tn_sq, w_sqrt_1, w_sqrt_2, u, sqrt_arg, ah, th;

    tn0_sq = t0 * t0 - c * h0 * h0;

    w_sqrt_1 = (h + h0) * (h + h0) - (mh - m0) * (mh - m0);
    w_sqrt_2 = (h - h0) * (h - h0) - (mh - m0) * (mh - m0);

    if (w_sqrt_1 < 0 || w_sqrt_2 < 0) {
        return NEGATIVE_SQUARED_ROOT;
    }

    u = sqrt(w_sqrt_1) + sqrt(w_sqrt_2);

    th = t0;
    if (fabs(h) > fabs(h0)) {

        if (!u) {
            return DIVISION_BY_ZERO;
        }

        sqrt_arg = c * h * h + 4 * h * h / (u * u) * tn0_sq;

        if (sqrt_arg < 0) {
            return NEGATIVE_SQUARED_ROOT;
        }

        th = sqrt(sqrt_arg);
    }
    else if (fabs(h) < fabs(h0)) {

        if (!h0) {
            return DIVISION_BY_ZERO;
        }

        sqrt_arg = c * h * h + u * u / (4 * h0 * h0) * tn0_sq;

        if (sqrt_arg < 0) {
            return NEGATIVE_SQUARED_ROOT;
        }

        th = sqrt(sqrt_arg);
    }

    tn_sq = th * th - c * h * h;

    if (!th || !tn0_sq) {
        return DIVISION_BY_ZERO;
    }

    ah = (t0 * tn_sq) / (th * tn0_sq) * slope;

    *out = th + ah * (m - mh);

    return NO_ERROR;
}

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
) {
    unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIndex < totalParameterCount) {
        unsigned int idxVelocity = (threadIndex / countSlope) % countVelocity;
        unsigned int idxSlope = threadIndex % countSlope;

        float v = minVelocity + static_cast<float>(idxVelocity) * incrementVelocity;
        float slope = minSlope + static_cast<float>(idxSlope) * incrementSlope;

        unsigned int offset = 2 * threadIndex;
        parameterArray[offset] = 4.0f / (v * v);
        parameterArray[offset + 1] = slope;
    }
}

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
) {
    unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int sampleIndex = threadIndex / totalParameterCount;
    unsigned int parameterIndex = threadIndex % totalParameterCount;
    unsigned int parameterOffset = 2 * parameterIndex;

    if (sampleIndex < samplesPerTrace) {
        enum gpu_error_code errorCode;

        float semblance = 0;
        float stack = 0;
        float mh, t;

        float t0 = static_cast<float>(sampleIndex) * dtInSeconds;

        float c = parameterArray[parameterOffset];
        float slope = parameterArray[parameterOffset + 1];

        float numeratorComponents[MAX_WINDOW_SIZE];
        float denominatorSum = 0;
        float linearSum = 0;

        for (unsigned int i = 0; i < MAX_WINDOW_SIZE; i++) {
            numeratorComponents[i] = 0;
        }

        unsigned int usedCount = 0, notUsedCount = 0;

        for (unsigned int traceIndex = 0; traceIndex < traceCount; traceIndex++) {
            float m = midpoint[traceIndex];
            float h = halfoffset[traceIndex];
            const float *traceSamples = samples + traceIndex * samplesPerTrace;

            errorCode = computeDisplacedMidpoint(h, h0, t0, m0, c, slope, &mh);

            if (errorCode == NO_ERROR && fabs(m - mh) <= apm) {
                notUsedCount++;
                continue;
            }

            errorCode = computeTime(h, h0, t0, m0, m, mh, c, slope, &t);

            if (errorCode == NO_ERROR) {
                float tIndex = t / dtInSeconds;
                int kIndex = static_cast<int>(tIndex);
                float dt = tIndex - static_cast<float>(kIndex);
            
                if ((kIndex - tauIndexDisplacement >= 0) &&
                    (kIndex + tauIndexDisplacement + 1 < static_cast<int>(samplesPerTrace))) {
                    
                    int k = kIndex - tauIndexDisplacement;
                    float u, y0, y1;
                    
                    y1 = traceSamples[k];
                    
                    for (int j = 0; j < windowSize; j++, k++) {
                        y0 = y1;
                        y1 = traceSamples[k + 1];
                        u = (y1 - y0) * dt + y0;
            
                        numeratorComponents[j] += u;
                        linearSum += u;
                        denominatorSum += u * u;
                    }
            
                    usedCount++;
                }
            }
        }

        if (usedCount > 0) {
            float sumNumerator = 0;
            for (int w = 0; w < windowSize; w++) {
                sumNumerator += numeratorComponents[w] * numeratorComponents[w];
            }

            semblance = sumNumerator / (usedCount * denominatorSum);
            stack = linearSum / (usedCount * windowSize);
        }

        notUsedCountArray[threadIndex] += notUsedCount;

        unsigned int offset = sampleIndex * totalParameterCount + parameterIndex;
        semblanceArray[offset] = semblance;
        stackArray[offset] = stack;
    }
}

__global__
void selectBestSemblancesForOffsetContinuationTrajectory(
    const float *semblanceArray,
    const float *stackArray,
    const float *parameterArray,
    unsigned int totalParameterCount,
    unsigned int samplesPerTrace,
    float *resultArray
) {
    unsigned int sampleIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (sampleIndex < samplesPerTrace) {
        unsigned int offset = sampleIndex * totalParameterCount;

        float bestSemblance = -1, bestStack, bestVelocity, bestSlope;

        for (unsigned int parameterIndex = 0; parameterIndex < totalParameterCount; parameterIndex++) {
            unsigned int offsetParameter = parameterIndex * 2;

            float semblance = semblanceArray[offset + parameterIndex];
            float stack = stackArray[offset + parameterIndex];
            float c = parameterArray[offsetParameter];
            float slope = parameterArray[offsetParameter + 1];

            if (semblance > bestSemblance) {
                bestSemblance = semblance;
                bestStack = stack;
                bestVelocity = 2 / sqrt(c);
                bestSlope = slope;
            }
        }

        resultArray[sampleIndex] = bestSemblance;
        resultArray[samplesPerTrace + sampleIndex] = bestStack;
        resultArray[2 * samplesPerTrace + sampleIndex] = bestVelocity;
        resultArray[3 * samplesPerTrace + sampleIndex] = bestSlope;
    }
}

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
) {
    unsigned int traceIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (traceIndex < traceCount) {

        float m, h, mh;
        m = midpointArray[traceIndex];
        h = halfoffsetArray[traceIndex];

        for (unsigned int sampleIndex = 0; !usedTraceMaskArray[traceIndex] && sampleIndex < samplesPerTrace; sampleIndex++) {
            float t0 = static_cast<float>(sampleIndex) * dtInSeconds;

            for (unsigned int parameterIndex = 0; !usedTraceMaskArray[traceIndex] && parameterIndex < parameterCount; parameterIndex++) {
                unsigned int offset = 2 * parameterIndex;

                float c = parameterArray[offset];
                float slope = parameterArray[offset + 1];

                enum gpu_error_code errorCode = computeDisplacedMidpoint(h, h0, t0, m0, c, slope, &mh);

                usedTraceMaskArray[traceIndex] = (errorCode == NO_ERROR && fabs(m - mh) <= apm);
            }
        }
    }
}