//
// Created by mikewu on 21-6-3.
//

#include "PillarScatter.h"
#include "plugin.h"
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define DEBUG 0

#define CHECK_CUDA(call) do {    \
  cudaError_t status = call; \
  if( status != cudaSuccess ) { \
    return status; \
  } \
} while(0)

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::PillarScatter;
using nvinfer1::plugin::PillarScatterPluginCreator;

namespace 
{
    const char* PILLAR_SCATTER_PLUGIN_VERSION{"1"};
    const char* PILLAR_SCATTER_PLUGIN_NAME{"PillarScatter"};
}

PluginFieldCollection PillarScatterPluginCreator::mFC{};
std::vector<PluginField> PillarScatterPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(PillarScatterPluginCreator);

PillarScatterPluginCreator::PillarScatterPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("grid_x", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("grid_y", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_features", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* PillarScatterPluginCreator::getPluginName() const noexcept
{
    return PILLAR_SCATTER_PLUGIN_NAME;
}

const char* PillarScatterPluginCreator::getPluginVersion() const noexcept
{
    return PILLAR_SCATTER_PLUGIN_VERSION;
}

const PluginFieldCollection* PillarScatterPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* PillarScatterPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "grid_x")) 
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
            grid_x_ = *(static_cast<const int*>(fields[i].data));
        } 
        else if(!strcmp(attrName, "grid_y")) 
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
            grid_y_ = *(static_cast<const int*>(fields[i].data));
        } 
        else if (!strcmp(attrName, "num_features")) 
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
            num_features_ = *(static_cast<const int*>(fields[i].data));
        }
    }
    return new PillarScatter(grid_x_, grid_y_, num_features_);
}

IPluginV2* PillarScatterPluginCreator::deserializePlugin(const char* name, const void* data, size_t length) noexcept
{
    return new PillarScatter(data, length);
}
void PillarScatterPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* PillarScatterPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

PillarScatter::PillarScatter(int grid_x, int grid_y, int num_features) 
    : grid_x_(grid_x)
    , grid_y_(grid_y)
    , num_features_(num_features)
{}

PillarScatter::PillarScatter(const void* buffer, size_t length) 
{
    const char *d = reinterpret_cast<const char*>(buffer), *a = d;
    grid_x_ = read<int>(d);
    grid_y_ = read<int>(d);
    num_features_ = read<int>(d);
    mInputDims = Dims3();
    mInputDims.d[0] = read<int>(d);
    mInputDims.d[1] = read<int>(d);
    mInputDims.d[2] = read<int>(d);
    mOutputDims = Dims3();
    mOutputDims.d[0] = read<int>(d);
    mOutputDims.d[1] = read<int>(d);
    mOutputDims.d[2] = read<int>(d);
    PLUGIN_ASSERT(d == a + length);
}

int PillarScatter::getNbOutputs() const noexcept
{
    return 1;
}

Dims PillarScatter::getOutputDimensions(int index, const Dims* inputDims, int nbInputs)  noexcept
{
    PLUGIN_ASSERT(index == 0);
    PLUGIN_ASSERT(inputDims);
    PLUGIN_ASSERT(nbInputs == 4);
    nvinfer1::Dims const& input = inputDims[0];
    nvinfer1::Dims output;
    output.nbDims = input.nbDims;
    // for (int d = 0; d < input.nbDims; ++d) {
    //     output.type[d] = input.type[d];
    // }
    output.d[0] = num_features_;
    output.d[1] = grid_y_;
    output.d[2] = grid_x_;
    return output;

}

int PillarScatter::initialize() noexcept
{
    return 0;
}

void PillarScatter::terminate() noexcept {}

void PillarScatter::destroy() noexcept
{
    delete this;
}

size_t PillarScatter::getWorkspaceSize(int) const noexcept
{
    return 0;
}

size_t PillarScatter::getSerializationSize() const noexcept
{
    return 3 * sizeof(int) + 3 * 2 * sizeof(int);
}

void PillarScatter::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, grid_x_);
    write(d, grid_y_);
    write(d, num_features_);
    write(d, mInputDims.d[0]);
    write(d, mInputDims.d[1]);
    write(d, mInputDims.d[2]);
    write(d, mOutputDims.d[0]);
    write(d, mOutputDims.d[1]);
    write(d, mOutputDims.d[2]);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

const char* PillarScatter::getPluginType() const noexcept
{ 
    return PILLAR_SCATTER_PLUGIN_NAME;
}

const char* PillarScatter::getPluginVersion() const noexcept
{
    return PILLAR_SCATTER_PLUGIN_VERSION;
}

IPluginV2Ext* PillarScatter::clone() const noexcept
{
    auto plugin = new PillarScatter(*this);
    plugin->setPluginNamespace(mNameSpace.c_str());
    return plugin;
}

void PillarScatter::setPluginNamespace(const char* libNamespace) noexcept
{
    mNameSpace = libNamespace;
}

const char* PillarScatter::getPluginNamespace() const  noexcept
{ 
    return mNameSpace.c_str();
}

bool PillarScatter::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR );
}

// Return the DataType of the plugin output at the requested index
DataType PillarScatter::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    PLUGIN_ASSERT(index <= 3);
    // Only DataType::kFLOAT is acceptable by the plugin layer
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool PillarScatter::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool PillarScatter::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

// Configure the layer with input and output data types.
void PillarScatter::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept
{
    PLUGIN_ASSERT(nbInputs == 4);
    mInputDims = inputDims[0];

    PLUGIN_ASSERT(nbOutputs == 1);
    mOutputDims = outputDims[0];
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
// void PillarScatter::attachToContext(
//     cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
// {
// }

// Detach the plugin object from its execution context.
// void PillarScatter::detachFromContext() {}

__global__ void scatterKernal(const int num_pillars, const int num_features, const int max_num_pillars,
                              const float* x_coord, const float* y_coord, const float* input, 
                              float* output, const int grid_x, const int grid_y) {
    const int i_pillar = blockIdx.x;
    const int i_feature = threadIdx.x;
    if (i_pillar >= num_pillars) return;
    int x_ind = static_cast<int>(x_coord[i_pillar]);
    int y_ind = static_cast<int>(y_coord[i_pillar]);
    int indices = y_ind * grid_x + x_ind;
    float feature = input[i_feature * max_num_pillars + i_pillar];
    output[i_feature * grid_y * grid_x + indices] = feature;
}

inline int PillarScatterLayer(cudaStream_t stream, const int num_pillars, const int num_features,
                              const int max_num_pillars,
                              const float* x_coord, const float* y_coord, const float* input,
                              float* output, const int grid_x, const int grid_y) {
    scatterKernal<<<num_pillars, num_features, 0, stream>>>(num_pillars, num_features, max_num_pillars, x_coord, 
                                                            y_coord, input, output, grid_x, grid_y);
    return cudaGetLastError() != cudaSuccess;
}



int PillarScatter::enqueue(int32_t batchSize, const void *const *inputs, void * const*outputs, void *workspace,
                  cudaStream_t stream) noexcept {

    const float* input_data = static_cast<const float*>(inputs[0]);
    const float* x_coord = static_cast<const float*>(inputs[1]);
    const float* y_coord = static_cast<const float*>(inputs[2]);
    const float* valid_pillar_count = static_cast<const float*>(inputs[3]);
    float num_pillars = 0;
    CHECK_CUDA(cudaMemcpy(&num_pillars, valid_pillar_count, sizeof(float), cudaMemcpyDeviceToHost));
    int num_valid_pillar = static_cast<int>(num_pillars);
    float* out_data = static_cast<float*>(outputs[0]);
    int max_num_pillars_ = mInputDims.d[1];
    CHECK_CUDA(cudaMemset(out_data, 0, grid_x_ * grid_y_ * num_features_ * sizeof(float)));
    PillarScatterLayer(stream, num_valid_pillar, num_features_, max_num_pillars_, x_coord, y_coord, input_data, out_data,
                       grid_x_, grid_y_);

    return cudaGetLastError() != cudaSuccess;
}
