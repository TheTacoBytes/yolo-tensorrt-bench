#pragma once
#include <cuda_runtime.h>
#ifdef __cplusplus
extern "C" {
#endif

void launchYoloNMS(
    const float* detInput,
    float*       detOutput,
    float        confThresh,
    float        iouThresh,
    int          maxDetections,
    int          HW,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
