#include "cuda/include/semblance/kernel/common/stretch_free.cuh"

 __global__
 void selectBestSemblances(
     const float *semblanceArray,
     const float *stackArray,
     const float *nArray,
     unsigned int nCount,
     unsigned int samplesPerTrace,
     float *resultArray
 ) {
     unsigned int sampleIndex = blockIdx.x * blockDim.x + threadIdx.x;
 
     if (sampleIndex < samplesPerTrace) {
         unsigned int offset = sampleIndex * nCount;
 
         float bestSemblance = -1, bestStack, bestN;
 
         for (unsigned int nIndex = 0; nIndex < nCount; nIndex++) {
             float semblance = semblanceArray[offset + nIndex];
             float stack = stackArray[offset + nIndex];
             float n = nArray[nIndex];
     
             if (semblance > bestSemblance) {
                 bestSemblance = semblance;
                 bestStack = stack;
                 bestN = n;
             }
         }
 
         resultArray[sampleIndex] = bestSemblance;
         resultArray[samplesPerTrace + sampleIndex] = bestStack;
         resultArray[2 * samplesPerTrace + sampleIndex] = bestN;
     }
 }
 