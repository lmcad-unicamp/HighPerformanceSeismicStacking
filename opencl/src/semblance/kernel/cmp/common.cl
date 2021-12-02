__kernel
void selectTracesForCommonMidPoint(
    __global float* midpointArray,
    unsigned int traceCount,
    __global char* usedTraceMaskArray,
    float m0
) {
    unsigned int traceIndex = get_group_id(0) * get_local_size(0) + get_local_id(0);

    if (traceIndex < traceCount) {
        usedTraceMaskArray[traceIndex] = (m0 == midpointArray[traceIndex]);
    }
}
