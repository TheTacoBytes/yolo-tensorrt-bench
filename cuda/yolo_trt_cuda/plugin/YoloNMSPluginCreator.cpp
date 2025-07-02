#include "YoloNMSPluginCreator.h"
#include "YoloNMSPlugin.h"
#include <cstring>

using namespace nvinfer1;

PluginFieldCollection YoloNMSPluginCreator::mFC{};
std::vector<PluginField>    YoloNMSPluginCreator::mPluginAttributes;

YoloNMSPluginCreator::YoloNMSPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back("conf_thresh", nullptr, PluginFieldType::kFLOAT32, 1);
    mPluginAttributes.emplace_back("iou_thresh",  nullptr, PluginFieldType::kFLOAT32, 1);
    mPluginAttributes.emplace_back("max_det",     nullptr, PluginFieldType::kINT32,   1);
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields   = mPluginAttributes.data();
}

IPluginV2* YoloNMSPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc) noexcept
{
    float conf=0.5f, iou=0.45f;
    int   maxDet=100;
    for (int i = 0; i < fc->nbFields; ++i) {
        auto& f = fc->fields[i];
        if (!strcmp(f.name, "conf_thresh")) conf   = *static_cast<float const*>(f.data);
        if (!strcmp(f.name, "iou_thresh"))  iou    = *static_cast<float const*>(f.data);
        if (!strcmp(f.name, "max_det"))     maxDet = *static_cast<int   const*>(f.data);
    }
    auto* plugin = new YoloNMSPlugin(conf, iou, maxDet);
    plugin->setPluginNamespace(mNamespace);
    return plugin;
}

IPluginV2* YoloNMSPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    auto* plugin = new YoloNMSPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace);
    return plugin;
}

PluginFieldCollection const* YoloNMSPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

// Register plugin creator
REGISTER_TENSORRT_PLUGIN(YoloNMSPluginCreator);
