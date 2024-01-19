import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import os
import torch
import numpy as np

def onnx2Engine(onnx_file_path, engine_file_path, patchsize, max_workspace_size, max_batch_size):
    TRT_LOGGER = trt.Logger()
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(explicit_batch)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    runtime = trt.Runtime(TRT_LOGGER)
    print("common.EXPLICIT_BATCH:", explicit_batch)

    config.max_workspace_size = max_workspace_size
    ##config.set_flag(trt.BuilderFlag.FP16)
    print("max_workspace_size:", config.max_workspace_size)
    builder.max_batch_size = max_batch_size

    if not os.path.exists(onnx_file_path):
        print(f"onnx file {onnx_file_path} not found")
        exit(0)
    print(f"Loading ONNX file from path {onnx_file_path}...")
    with open(onnx_file_path, "rb") as model:
        print("Beginning ONNX file parsing")
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print("inputs:jdaksj")
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    print("inputs:", inputs)

    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    print("outputs:", outputs)
    print("Network Description")

    for input in inputs:
        batch_size = input.shape[0]
        print("Input '{}' with shape {} and dtype {} .".format(input.name, input.shape, input.dtype))
    for output in outputs:
        print("output '{}' with shape {} and dtype {} .".format(output.name, output.shape, output.dtype))

    # Dynamic input setting 动态输入在builder的profile设置
    # 为每个动态输入绑定一个profile
    """
    profile = builder.create_optimization_profile()
    print("network.get_input(0).name:", network.get_input(0).name)
    profile.set_shape(network.get_input(0).name, (1, *patchsize), (1, *patchsize),
                      (max_batch_size, *patchsize))  # 最小的尺寸,常用的尺寸,最大的尺寸,推理时候输入需要在这个范围内
    config.add_optimization_profile(profile)
    """
    engine = builder.build_serialized_network(network, config)
    print('Completed creating Engine')
    with open(engine_file_path, 'wb') as f:
        f.write(engine)
    return engine

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def test_engine(engine_file, x):
    trt_logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_file, 'rb') as f:
    #with open("tinyengine.engine", 'rb') as f:
        engine_data = f.read()
    runtime = trt.Runtime(trt_logger)
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()
    assert context
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        if engine.binding_is_input(binding):
            data = np.ones(size, dtype = dtype)
            np.copyto(host_mem, data.ravel())
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    import pdb;pdb.set_trace()
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]
    # 处理输出结果...

if __name__ == "__main__":
    ##engine_trt = onnx2Engine("./tinynet.onnx", "tinyengine.engine", [3, 12, 12], 5*(1<<30), 2)

    ipt = torch.ones(2, 3, 12, 12).cuda()
    outputs = test_engine("tinyengine.engine", ipt)
    import pdb;pdb.set_trace()
    output_cpu = np.frombuffer(out_puts[0], dtype= np.float32).reshape(1,1,12,12)
    print("jdalkjdlk")
