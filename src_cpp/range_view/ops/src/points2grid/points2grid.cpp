#include<torch/extension.h>
#include<torch/serialize/tensor.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x, "must be a CUDAtensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, "must be a cotiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x) CHECK_CONTIGUOUS(x)

void points2grid_launcher(float* points, int* indices, float* grid, int n, int n_channels);

void points2grid_gpu(at::Tensor points_tensor, at::Tensor indices_tensor, at::Tensor grid_tensor){
	CHECK_INPUT(points_tensor);
	CHECK_INPUT(indices_tensor);
	CHECK_INPUT(grid_tensor);
	
	float* points = points_tensor.data_ptr<float>();
	int* indices = indices_tensor.data_ptr<int>();
    float* grid = grid_tensor.data_ptr<float>();
	int n = points_tensor.size(1);
    int n_channels = points_tensor.size(0);
	points2grid_launcher(points, indices, grid, n, n_channels);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &points2grid_gpu, "points to grid (CUDA)");
}
