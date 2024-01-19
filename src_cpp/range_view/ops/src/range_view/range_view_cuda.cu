#include <cstdio>
#include <cassert>
#include <math.h>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define DIVUP(m, n) ((m + n - 1) / n) 
#define PI 3.1415926

__global__ void range_view_kernel(const float* points, int* coors, int n){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// int tid = threadIdx.x;
	if (idx >= n){
		return ;
	}
	float p_x = points[idx * 3];
	float p_y = points[idx * 3 + 1];
	float p_z = points[idx * 3 + 2];
	float degree_h = asin(p_y / sqrt(p_y * p_y + p_x * p_x)) * 180 / PI;
	float degree_v = asin(p_z / sqrt(p_y * p_y + p_x * p_x + p_z * p_z)) * 180 / PI;
	if (p_x < 0) {
		degree_h = 180 - degree_h;
	}
	degree_h = degree_h + 90;
	coors[idx * 2] =  static_cast<int>(degree_h / 0.1) % 3600;
	coors[idx * 2 + 1] =  static_cast<int>((degree_v + 50) / 0.2) % 500;
	//coors[idx * 2 + 1] =  static_cast<int>((degree_v + 12.47) / 0.2);
	return ;
}
void range_view_launcher(const float* points, int* coors, int n){
	dim3 blockSize(DIVUP(n, THREADS_PER_BLOCK));
	dim3 threadSize(THREADS_PER_BLOCK);
	range_view_kernel<<<blockSize, threadSize>>>(points, coors, n);
}
