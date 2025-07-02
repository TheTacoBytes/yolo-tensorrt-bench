#include <cuda_runtime.h>

//-----------------------------------------------------
// compute IoU between two boxes in [x,y,w,h] center form
//-----------------------------------------------------
__device__ float iou(const float *a, const float *b) {
    float ax = a[0], ay = a[1], aw = a[2], ah = a[3];
    float bx = b[0], by = b[1], bw = b[2], bh = b[3];

    float ax1 = ax - aw*0.5f, ay1 = ay - ah*0.5f;
    float ax2 = ax + aw*0.5f, ay2 = ay + ah*0.5f;
    float bx1 = bx - bw*0.5f, by1 = by - bh*0.5f;
    float bx2 = bx + bw*0.5f, by2 = by + bh*0.5f;

    float ix1 = max(ax1, bx1), iy1 = max(ay1, by1);
    float ix2 = min(ax2, bx2), iy2 = min(ay2, by2);
    float iw  = max(0.0f, ix2 - ix1);
    float ih  = max(0.0f, iy2 - iy1);
    float inter = iw * ih;
    float areaA = aw * ah;
    float areaB = bw * bh;
    return inter / (areaA + areaB - inter);
}

//-----------------------------------------------------------------------------
// Kernel: performs NMS on up to HW detections, writes at most maxDet outputs
// detInput:  [HW][6] float = x,y,w,h,conf,cls
// detOutput: [maxDet][6]
//-----------------------------------------------------------------------------
__global__ void yoloNmsKernel(
    const float* detInput,
    float*       detOutput,
    float        confThresh,
    float        iouThresh,
    int          maxDet,
    int          HW)
{
    // we’ll do the entire NMS in thread 0 of this block
    if (threadIdx.x > 0) return;

    int outCount = 0;

    // loop over every proposal
    for (int i = 0; i < HW; i++) {
        const float* d = detInput + i*6;
        float conf = d[4];
        if (conf < confThresh) continue;

        // compare against all already-kept boxes
        bool keep = true;
        for (int k = 0; k < outCount; k++) {
            float* kept = detOutput + k*6;
            if (iou(kept, d) > iouThresh) {
                keep = false;
                break;
            }
        }
        if (!keep) continue;

        // accept it
        if (outCount < maxDet) {
            float* out = detOutput + outCount*6;
            out[0] = d[0];
            out[1] = d[1];
            out[2] = d[2];
            out[3] = d[3];
            out[4] = d[4];
            out[5] = d[5];
            outCount++;
        }
    }

    // zero-pad the rest so downstream knows there’s no more
    for (int k = outCount; k < maxDet; k++) {
        float* out = detOutput + k*6;
        out[0]=out[1]=out[2]=out[3]=out[4]=out[5]=0.0f;
    }
}

//------------------------------------------------------------------------------
// host wrapper — called from your plugin’s enqueue()
//------------------------------------------------------------------------------
void launchYoloNMS(
    const float* detInput, float* detOutput,
    float confThresh, float iouThresh, int maxDet, int HW,
    cudaStream_t stream)
{
  yoloNmsKernel<<<1,256,0,stream>>>(
      detInput, detOutput, confThresh, iouThresh, maxDet, HW);
}