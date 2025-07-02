#pragma once
#include <cuda_runtime.h>
#include <NvInfer.h>

class YoloNMSPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    // ctor for building from parameters
    YoloNMSPlugin(float confThresh, float iouThresh, int maxDetections);
    // deserialization ctor
    YoloNMSPlugin(void const* data, size_t length);

    // IPluginV2
    char const* getPluginType() const noexcept override { return "YoloNMS"; }
    char const* getPluginVersion() const noexcept override { return "1"; }
    int getNbOutputs() const noexcept override { return 1; }
    void destroy() noexcept override { delete this; }

    // IPluginV2Ext / IPluginV2
    int32_t initialize() noexcept override { return 0; }
    void    terminate() noexcept override {}
    nvinfer1::DataType getOutputDataType(
        int outputIndex,
        const nvinfer1::DataType* inputTypes, int nbInputs
    ) const noexcept override { return nvinfer1::DataType::kFLOAT; }

    // IPluginV2DynamicExt
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex,
        nvinfer1::DimsExprs const* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder
    ) noexcept override;
    bool supportsFormatCombination(
        int pos,
        nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs
    ) noexcept override;
    void configurePlugin(
        nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs
    ) noexcept override;
    size_t getWorkspaceSize(
        nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int nbOutputs
    ) const noexcept override;
    int enqueue(
        nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream
    ) noexcept override;

    // serialization
    void serialize(void* buffer) const noexcept override;
    size_t getSerializationSize() const noexcept override;

    // namespace
    void setPluginNamespace(char const* libNamespace) noexcept override { mNamespace = libNamespace; }
    char const* getPluginNamespace() const noexcept override { return mNamespace; }

private:
    float mConfThresh, mIouThresh;
    int   mMaxDetections;
    int   mInputC, mInputHW;
    const char* mNamespace{""};

    void deserialize(void const* data, size_t length);
};
