//
// Created by mikewu on 21-6-3.
//

#ifndef PILLAR_SCATTER_PLUGIN_H
#define PILLAR_SCATTER_PLUGIN_H

#include "NvInfer.h"
#include "plugin.h"
#include <vector>
#include <cuda_runtime.h>
namespace nvinfer1
{
namespace plugin
{

class PillarScatter : public nvinfer1::IPluginV2Ext 
{
public:
    PillarScatter(int grid_x, int grid_y, int num_features);

    PillarScatter(const void* buffer, size_t length);

    ~PillarScatter() override = default;

    int getNbOutputs() const noexcept override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept  override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    void destroy() noexcept override;

    size_t getWorkspaceSize(int maxBatchSize) const noexcept override;

    int enqueue(
        int32_t batchSize, const void* const* inputs, void*const* outputs, void* workspace, cudaStream_t stream) noexcept  override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;

    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    IPluginV2Ext* clone() const noexcept override;

    void setPluginNamespace(const char* libNamespace)noexcept override;

    const char* getPluginNamespace() const noexcept  override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept  override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;

    // void attachToContext(
    //     cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept  override;

    // void detachFromContext() override;

private:
    int grid_x_, grid_y_, num_features_;
    Dims mInputDims;
    Dims mOutputDims;
    std::string mNameSpace;
};

class PillarScatterPluginCreator :  public nvinfer1::IPluginCreator
{
public:
    PillarScatterPluginCreator();

    ~PillarScatterPluginCreator() {};

    const char* getPluginName() const noexcept  override;

    const char* getPluginVersion() const noexcept  override;

    const PluginFieldCollection* getFieldNames() noexcept  override; 

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept  override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept  override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    int grid_x_, grid_y_, num_features_;
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif //PILLAR_SCATTER_PLUGIN_H
