#pragma once
#include <NvInfer.h>
#include <vector>

namespace nvinfer1 {

class YoloNMSPluginCreator : public IPluginCreator {
public:
    YoloNMSPluginCreator();

    // IPluginCreator Methods
    const char* getPluginName() const noexcept override { return "YoloNMS"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const PluginFieldCollection* getFieldNames() noexcept override;
    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* libNamespace) noexcept override { mNamespace = libNamespace; }
    const char* getPluginNamespace() const noexcept override { return mNamespace; }

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    const char* mNamespace{""};
};

}
