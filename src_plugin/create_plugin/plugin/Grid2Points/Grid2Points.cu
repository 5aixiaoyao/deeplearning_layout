/**
 * For the usage of those member function, please refer to the
 * offical api doc.
 * https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_plugin_v2_ext.html
 */


#include <cassert>
#include <iostream>
#include <string.h>

#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "Grid2Points.h"

// Use fp16 mode for inference
//#define DATA_TYPE nvinfer1::DataType::kHALF
#define DATA_TYPE nvinfer1::DataType::kFLOAT
#define THREAD_NUM 1024

#define CHECK_CUDA(call) do {    \
  cudaError_t status = call; \
  if( status != cudaSuccess ) { \
    return status; \
  } \
} while(0)

// Helper function for deserializing plugin
template <typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}
// Helper function for serializing plugin
template <typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

using namespace nvinfer1;
using nvinfer1::plugin::Grid2PointsPlugin;
using nvinfer1::plugin::Grid2PointsPluginCreator;

static const char* SCATTERND_PLUGIN_VERSION{"1"};
static const char* SCATTERND_PLUGIN_NAME{"Grid2PointsPlugin"};

PluginFieldCollection Grid2PointsPluginCreator::mFC{};
std::vector<PluginField> Grid2PointsPluginCreator::mPluginAttributes;

Grid2PointsPlugin::Grid2PointsPlugin(const std::string name, int channels, int features_num, const DataType type) : mLayerName(name), mDataType(type)
{
    mChannels = channels;
    mFeatures_num = features_num;
    // mOutputSize[0] = outputShapeArray[0];
    // mOutputSize[1] = outputShapeArray[1];

    // mInputIndexSize[0] = indexShapeArray[0];
    // mInputIndexSize[1] = indexShapeArray[1];

}

Grid2PointsPlugin::Grid2PointsPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    const char *d = reinterpret_cast<const char *>(data);
    const char *a = d;

    mDataType = readFromBuffer<DataType>(d);
    mChannels = readFromBuffer<int>(d);
    mFeatures_num = readFromBuffer<int>(d);
    // mOutputSize[0] = readFromBuffer<size_t>(d);
    // mOutputSize[1] = readFromBuffer<size_t>(d);
    // mInputIndexSize[0] = readFromBuffer<size_t>(d);
    // mInputIndexSize[1] = readFromBuffer<size_t>(d);

    assert(d == a + length);
}

int Grid2PointsPlugin::getNbOutputs() const noexcept
{
    return 1;
}

Dims Grid2PointsPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{   
    // scatterND data input 
    printf("%d\n", inputs[1].d[0]);
    return Dims2(inputs[1].d[2], inputs[2].d[1]);
}

int Grid2PointsPlugin::initialize() noexcept
{
    return 0;
}

size_t Grid2PointsPlugin::getWorkspaceSize(int) const noexcept
{
    return 0;
}

DataType Grid2PointsPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    return inputTypes[2];
}

// template <typename Dtype>
// __global__ void _ScatterNDKernel(const int *points_num, const int *indicesInputPtr , const Dtype* input, Dtype* output, int channel_num) {
    
//     int idx_num = blockDim.x * blockIdx.x + threadIdx.x;
//     if (idx_num >= *points_num) return;    
    
//     int idx_output = indicesInputPtr[idx_num];
//     if (idx_output < 0) return;

//     // while(atomicCAS(&flag[idx_output], 0, 1) == 1){};
//     // flag[idx_output] = 1;
//     for(int idx=0; idx < channel_num; idx++){
//         if (output[idx_output*channel_num+idx] < updata_input[idx_num*channel_num+idx]){
//             output[idx_output*channel_num+idx] = updata_input[idx_num*channel_num+idx];
//         }
//         //output[idx_output*channel_num+idx] = updata_input[idx_num*channel_num+idx];
//     }
//     // atomicExch(&flag[idx_output], 0);
// }
template <typename Dtype>
__global__ void grid2points_kernel(const int *points_num, const int *indices, const Dtype* grid, Dtype* points, int n_channels, int mFeatures_num){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// int tid = threadIdx.x;
	if (idx >= *points_num){
		return ;
	}
    for (int i_channels = 0; i_channels < n_channels; i_channels++){
        if (indices[idx] > mFeatures_num - 1){
            continue;
        }
        // points[idx * n_channels + i_channels] = 1;
        points[idx * n_channels + i_channels] = grid[indices[idx] * n_channels + i_channels];
        // printf("points[idx + i_channels]: %f\n", grid[indices[idx] * n_channels + i_channels]);
    }
	return ;
}

int Grid2PointsPlugin::enqueue(int batchSize, const void* const* inputs, void* const* outputs, void*, cudaStream_t stream)noexcept
{
    int channel_num = mChannels;
    const int* valid_number_count = static_cast<const int*>(inputs[0]);
    int max_index_num = 0;
    CHECK_CUDA(cudaMemcpy(&max_index_num, valid_number_count, sizeof(int), cudaMemcpyDeviceToHost));
    // int max_index_num = mFeatures_num;

    int totalElems = max_index_num * channel_num;
    
    dim3 blockSize(THREAD_NUM);
    dim3 gridsize(max_index_num / blockSize.x + 1);

    // if you want to inference use fp32, change the DATA_TYPE
    printf("channel_num:%d\n", channel_num);
    printf("blockSize.x:%d\n", blockSize.x);
    printf("max_index_num:%d\n", max_index_num);
    printf("gridsize:%d\n", gridsize.x);
    switch (mDataType)
    {
    case nvinfer1::DataType::kFLOAT:
        cudaMemset(outputs[0], 0, totalElems * sizeof(float));
        // printf("outputs:%f\n",);
        grid2points_kernel<<<gridsize, blockSize,0,stream>>>(static_cast<int32_t const*> (inputs[0]), static_cast<int32_t const*> (inputs[1]), static_cast<float const*> (inputs[2]), 
                                                    static_cast<float *> (outputs[0]), channel_num, mFeatures_num);
        break;

    case nvinfer1::DataType::kHALF:
        cudaMemset(outputs[0], 0, totalElems * sizeof(float)/2);
        grid2points_kernel<<<gridsize, blockSize,0,stream>>>(static_cast<int32_t const*> (inputs[0]), static_cast<int32_t const*> (inputs[1]), static_cast<float const*> (inputs[2]),
                                                    static_cast<float *> (outputs[0]), channel_num, mFeatures_num);
        // _ScatterNDKernel<<<gridsize, blockSize,0,stream>>>(static_cast<int16_t const*> (inputs[2]), static_cast<int32_t const*> (inputs[1]), 
        //                                             static_cast<int16_t *> (outputs[0]), channel_num, max_index_num);
        
        break;
    
    default:
        std::cout << "[ERROR]: mDataType dones't support" << std::endl;
    }
    return 0;
}

void Grid2PointsPlugin::serialize(void* buffer) const noexcept
{
    char* d = static_cast<char*>(buffer);
    char *a = d;
    writeToBuffer<DataType>(d, mDataType);
    writeToBuffer<int>(d, mChannels);
    writeToBuffer<int>(d, mFeatures_num);
    // writeToBuffer<size_t>(d, mOutputSize[0]);
    // writeToBuffer<size_t>(d, mOutputSize[1]);
    // writeToBuffer<size_t>(d, mInputIndexSize[0]);
    // writeToBuffer<size_t>(d, mInputIndexSize[1]);

    assert(d == a + getSerializationSize());
}

void Grid2PointsPlugin::terminate() noexcept{
}

size_t Grid2PointsPlugin::getSerializationSize() const noexcept
{
    return sizeof(DataType)+ 2*sizeof(int);
}

bool Grid2PointsPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
{
    return false;
}

bool Grid2PointsPlugin::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

void Grid2PointsPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept
{
    // mOutputSize[0] = outputDims[0].d[0];
    // mOutputSize[1] = outputDims[0].d[1];
    // mInputIndexSize[0] = inputDims[0].d[0];
    // mInputIndexSize[1] = inputDims[1].d[1];
}

bool Grid2PointsPlugin::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    switch (type)
    {   
        case nvinfer1::DataType::kINT32: return true;
        case nvinfer1::DataType::kFLOAT: return true;
        case nvinfer1::DataType::kHALF: return true;
    }
    return false;
}

/**
 * NO NEED TO MODIFY
 */
const char* Grid2PointsPlugin::getPluginType() const noexcept
{
    return SCATTERND_PLUGIN_NAME;
}

/**
 * NO NEED TO MODIFY
 */
const char* Grid2PointsPlugin::getPluginVersion() const noexcept
{
    return SCATTERND_PLUGIN_VERSION;
}

void Grid2PointsPlugin::destroy()noexcept
{
    delete this;
}

IPluginV2Ext* Grid2PointsPlugin::clone() const noexcept
{
    auto plugin = new Grid2PointsPlugin(mLayerName, mChannels, mFeatures_num, mDataType);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

/**
 * NO NEED TO MODIFY
 */
void Grid2PointsPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

/**
 * NO NEED TO MODIFY
 */
const char* Grid2PointsPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

Grid2PointsPluginCreator::Grid2PointsPluginCreator()
{   
    mPluginAttributes.emplace_back(PluginField("channels", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("features_num", nullptr, PluginFieldType::kINT32, 1));
    // mPluginAttributes.emplace_back(PluginField("index_shape", nullptr, PluginFieldType::kINT32, 3));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

/**
 * NO NEED TO MODIFY
 */
const char* Grid2PointsPluginCreator::getPluginName() const noexcept
{
    return SCATTERND_PLUGIN_NAME;
}

/**
 * NO NEED TO MODIFY
 */
const char* Grid2PointsPluginCreator::getPluginVersion() const noexcept
{
    return SCATTERND_PLUGIN_VERSION;
}

/**
 * NO NEED TO MODIFY
 */
const PluginFieldCollection* Grid2PointsPluginCreator::getFieldNames() noexcept
{   
    return &mFC;
}

IPluginV2Ext* Grid2PointsPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)noexcept
{
    
    const nvinfer1::PluginField* fields = fc->fields;
    
    mDataType = DATA_TYPE;
    int channels = 64;
    int features_num = 100000;
    // size_t indexShapeArray[2] = {0};
    // size_t outputShapeArray[2] = {0};

    for (int i=0; i<fc->nbFields; i++) {
        if(!strcmp(fields[i].name, "channels")){
            const auto *channels_attr = static_cast<const int32_t*>(fields[i].data);
            channels = *channels_attr;
            // outputShapeArray[0] = outputShapeAttr[1];
            // outputShapeArray[1] = outputShapeAttr[2];

        }
        if(!strcmp(fields[i].name, "features_num")){
            const auto * features_num_attr = static_cast<const int32_t*>(fields[i].data);
            features_num = *features_num_attr;
            // indexShapeArray[0] = indexShapeAttr[1];
            // indexShapeArray[1] = indexShapeAttr[2];
        }
    }
    
    auto* plugin = new Grid2PointsPlugin(name, channels, features_num, mDataType);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2Ext* Grid2PointsPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{   
    return new Grid2PointsPlugin(name, serialData, serialLength);
}

REGISTER_TENSORRT_PLUGIN(Grid2PointsPluginCreator);

