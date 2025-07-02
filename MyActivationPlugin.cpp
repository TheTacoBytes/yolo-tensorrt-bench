#include <NvInfer.h>
#include "NvInferPlugin.h"
#include <cuda_runtime_api.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <vector>
#include <string>

using namespace nvinfer1;

//------------ Plugin Implementation ------------
class MyActivationPlugin : public IPluginV2DynamicExt
{
    float alpha_, beta_;
    std::string namespace_;

public:
    // Constructor for explicit creation
    MyActivationPlugin(float alpha, float beta)
        : alpha_(alpha), beta_(beta) {}

    // Deserialize constructor
    MyActivationPlugin(const void* data, size_t length)
    {
        const float* d = reinterpret_cast<const float*>(data);
        alpha_ = d[0];
        beta_  = d[1];
    }

    // IPluginV2DynamicExt overrides
    IPluginV2DynamicExt* clone() const noexcept override
    {
        return new MyActivationPlugin(alpha_, beta_);
    }

    DimsExprs getOutputDimensions(
        int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept override
    {
        return inputs[0];
    }

    bool supportsFormatCombination(
        int pos, const PluginTensorDesc* inOut, int nbInputsOutputs) noexcept override
    {
        return inOut[pos].type == DataType::kFLOAT
            && inOut[pos].format == TensorFormat::kLINEAR;
    }

    void configurePlugin(
        const DynamicPluginTensorDesc* in, int nbInputs,
        const DynamicPluginTensorDesc* out, int nbOutputs) noexcept override {}

    size_t getWorkspaceSize(
        const PluginTensorDesc* /*inputs*/, int /*nbInputs*/,
        const PluginTensorDesc* /*outputs*/, int /*nbOutputs*/) const noexcept override
    {
        return 0;
    }

    int enqueue(
        const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override
    {
        int volume = 1;
        for (int i = 0; i < inputDesc[0].dims.nbDims; i++)
            volume *= inputDesc[0].dims.d[i];

        const float* inData  = static_cast<const float*>(inputs[0]);
        float*       outData = static_cast<float*>(outputs[0]);

        thrust::transform(
            thrust::cuda::par.on(stream),
            inData, inData + volume, outData,
            [=] __device__ (float x) { return alpha_ * x + beta_; }
        );
        return 0;
    }

    DataType getOutputDataType(
        int index, const DataType* inputTypes, int nbInputs) const noexcept override
    {
        return inputTypes[0];
    }

    const char* getPluginType() const noexcept override   { return "MyActivationPlugin"; }
    const char* getPluginVersion() const noexcept override{ return "1"; }
    int getNbOutputs() const noexcept override            { return 1; }

    size_t getSerializationSize() const noexcept override
    {
        return 2 * sizeof(float);
    }

    void serialize(void* buffer) const noexcept override
    {
        float* d = reinterpret_cast<float*>(buffer);
        d[0] = alpha_;
        d[1] = beta_;
    }

    void destroy() noexcept override { delete this; }

    void setPluginNamespace(const char* libNamespace) noexcept override
    {
        namespace_ = libNamespace;
    }

    const char* getPluginNamespace() const noexcept override
    {
        return namespace_.c_str();
    }

    // Optional but recommended
    int initialize() noexcept override   { return 0; }
    void terminate() noexcept override   {}
    void attachToContext(
      cudnnContext* /*cudnn*/, cublasContext* /*cublas*/, IGpuAllocator* /*alloc*/) noexcept override {}
    void detachFromContext() noexcept override {}
};

//------------ Plugin Creator ------------
class MyActivationPluginCreator : public IPluginCreator
{
    PluginFieldCollection mFC{};
    std::vector<PluginField> mPluginAttributes;
    std::string namespace_;

public:
    MyActivationPluginCreator()
    {
        mPluginAttributes.emplace_back(PluginField{"alpha", nullptr, PluginFieldType::kFLOAT32, 1});
        mPluginAttributes.emplace_back(PluginField{"beta",  nullptr, PluginFieldType::kFLOAT32, 1});
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields   = mPluginAttributes.data();
    }

    const char* getPluginName() const noexcept override   { return "MyActivationPlugin"; }
    const char* getPluginVersion() const noexcept override{ return "1"; }
    const PluginFieldCollection* getFieldNames() noexcept override { return &mFC; }

    IPluginV2* createPlugin(
      const char* name, const PluginFieldCollection* fc) noexcept override
    {
        float alpha = 1.0f, beta = 0.0f;
        for (int i = 0; i < fc->nbFields; i++) {
            const auto& field = fc->fields[i];
            if (strcmp(field.name, "alpha") == 0)
                alpha = *static_cast<const float*>(field.data);
            else if (strcmp(field.name, "beta") == 0)
                beta  = *static_cast<const float*>(field.data);
        }
        auto* plugin = new MyActivationPlugin(alpha, beta);
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    }

    IPluginV2* deserializePlugin(
      const char* name, const void* serialData, size_t serialLength) noexcept override
    {
        auto* plugin = new MyActivationPlugin(serialData, serialLength);
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    }

    void setPluginNamespace(const char* libNamespace) noexcept override
    {
        namespace_ = libNamespace;
    }

    const char* getPluginNamespace() const noexcept override
    {
        return namespace_.c_str();
    }
};

// Register the plugin creator with TensorRT
REGISTER_TENSORRT_PLUGIN(MyActivationPluginCreator);
