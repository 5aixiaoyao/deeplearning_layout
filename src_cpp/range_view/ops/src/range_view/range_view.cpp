#include<torch/extension.h>
#include<torch/serialize/tensor.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x, "must be a CUDAtensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, "must be a cotiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x) CHECK_CONTIGUOUS(x)

void range_view_launcher(const float* points, int* coors, int n);

void range_view_gpu(at::Tensor points_tensor, at::Tensor coors_tensor){
	CHECK_INPUT(points_tensor);
	CHECK_INPUT(coors_tensor);
	
	const float* points = points_tensor.data_ptr<float>();
	int* coors = coors_tensor.data_ptr<int>();
	int n = points_tensor.size(0);
	range_view_launcher(points, coors, n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &range_view_gpu, "sum an array (CUDA)");
}
