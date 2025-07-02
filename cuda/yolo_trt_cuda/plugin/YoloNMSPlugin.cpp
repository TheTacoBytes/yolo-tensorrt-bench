#include "YoloNMSPlugin.h"
#include "yolo_nms_kernel.h"   
#include <cstring>
#include <cassert>

// ctor
YoloNMSPlugin::YoloNMSPlugin(float conf, float iou, int maxDet)
 : mConfThresh(conf), mIouThresh(iou), mMaxDetections(maxDet)
{}

// deserialization
YoloNMSPlugin::YoloNMSPlugin(void const* data, size_t length)
{
    deserialize(data,length);
}

void YoloNMSPlugin::deserialize(void const* data, size_t /*length*/)
{
    auto const* d = static_cast<char const*>(data);
    std::memcpy(&mConfThresh, d, sizeof(mConfThresh)); d += sizeof(mConfThresh);
    std::memcpy(&mIouThresh,  d, sizeof(mIouThresh));  d += sizeof(mIouThresh);
    std::memcpy(&mMaxDetections, d, sizeof(mMaxDetections));
}

nvinfer1::IPluginV2DynamicExt* YoloNMSPlugin::clone() const noexcept
{
    auto* p = new YoloNMSPlugin(mConfThresh,mIouThresh,mMaxDetections);
    p->setPluginNamespace(mNamespace);
    return p;
}

// output dims = [maxDetections x 6]
nvinfer1::DimsExprs YoloNMSPlugin::getOutputDimensions(
    int, nvinfer1::DimsExprs const*, int,
    nvinfer1::IExprBuilder& expr) noexcept
{
    nvinfer1::DimsExprs out;
    out.nbDims = 2;
    out.d[0]    = expr.constant(mMaxDetections);
    out.d[1]    = expr.constant(6);
    return out;
}

// only FP32 linear
bool YoloNMSPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* io,
    int, int) noexcept
{
    return io[pos].format == nvinfer1::TensorFormat::kLINEAR
        && io[pos].type   == nvinfer1::DataType::kFLOAT;
}

// grab input channels & HW
void YoloNMSPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* in, int,
    nvinfer1::DynamicPluginTensorDesc const*, int) noexcept
{
    auto const& d = in[0].desc.dims;
    mInputC  = d.d[1];
    mInputHW = d.d[2];
}

// no extra workspace
size_t YoloNMSPlugin::getWorkspaceSize(
    nvinfer1::PluginTensorDesc const*, int,
    nvinfer1::PluginTensorDesc const*, int) const noexcept
{
    return 0;
}

// call your CUDA NMS kernel
int YoloNMSPlugin::enqueue(
    nvinfer1::PluginTensorDesc const*, nvinfer1::PluginTensorDesc const*,
    void const* const* inputs, void* const* outputs,
    void*, cudaStream_t stream) noexcept
{
    const float* detInput  = static_cast<const float*>(inputs[0]);
    float*       detOutput = static_cast<float*>(outputs[0]);
    launchYoloNMS(
      detInput, detOutput,
      mConfThresh, mIouThresh, mMaxDetections,
      mInputHW, stream);
    return 0;
}

// serialization = three scalars
size_t YoloNMSPlugin::getSerializationSize() const noexcept
{
    return sizeof(mConfThresh)+sizeof(mIouThresh)+sizeof(mMaxDetections);
}
void YoloNMSPlugin::serialize(void* buffer) const noexcept
{
    auto* d = static_cast<char*>(buffer);
    std::memcpy(d, &mConfThresh,    sizeof(mConfThresh));    d += sizeof(mConfThresh);
    std::memcpy(d, &mIouThresh,     sizeof(mIouThresh));     d += sizeof(mIouThresh);
    std::memcpy(d, &mMaxDetections, sizeof(mMaxDetections));
}
