#ifndef __GRID2POINTS_LIDAR_PLUGIN__
#define __GRID2POINTS_LIDAR_PLUGIN__

#include "NvInferPlugin.h"
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{
class Grid2PointsPlugin : public IPluginV2Ext
{
public:
    Grid2PointsPlugin(const std::string name, int channels, int features_num, const DataType type);

    Grid2PointsPlugin(const std::string name, const void* data, size_t length);

    Grid2PointsPlugin() = delete;

    int getNbOutputs() const noexcept override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override;

    int initialize() noexcept override;
    void terminate() noexcept override;

    size_t getWorkspaceSize(int) const noexcept override;

    int enqueue(
        int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;

    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept override;

    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;

    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    IPluginV2Ext* clone() const noexcept override;

    void setPluginNamespace(const char* libNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    const std::string mLayerName;
    size_t mCopySize;
    std::string mNamespace;
    DataType mDataType;
    int mChannels;
    int mFeatures_num;
    // size_t mOutputSize[2]; // [H*W, C]
    // size_t mInputIndexSize[2]; // [H*W, C]
    // int flag[480*800] = {0};

};

class Grid2PointsPluginCreator : public IPluginCreator
{
public:
    Grid2PointsPluginCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    const char* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
    DataType mDataType;
};

} // namespace plugin
} // namespace nvinfer1


#endif
