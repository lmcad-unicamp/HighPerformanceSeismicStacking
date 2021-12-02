 #include "common/include/gpu/interface.h"

 __kernel
 void selectBestSemblances(
     __global float *semblanceArray,
     __global float *stackArray,
     __global float *nArray,
     unsigned int nCount,
     unsigned int samplesPerTrace,
     __global float *resultArray
 ) {
     unsigned int sampleIndex = get_group_id(0) * get_local_size(0) + get_local_id(0);

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
