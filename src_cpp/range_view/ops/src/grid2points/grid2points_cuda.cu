#include <cstdio>
#include <cassert>
#include <math.h>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define DIVUP(m, n) ((m + n - 1) / n) 
// #define PI 3.1415926

__global__ void grid2points_kernel(float* points, int* indices, float* grid, int n, int n_channels){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// int tid = threadIdx.x;
	if (idx >= n){
		return ;
	}
    for (int i_channels = 0; i_channels < n_channels; i_channels++){
        points[idx * n_channels + i_channels] = grid[indices[idx] * n_channels + i_channels];
        // printf("points[idx + i_channels]: %f\n", grid[indices[idx] * n_channels + i_channels]);
    }
	return ;
}
void grid2points_launcher(float* points, int* indices, float* grid, int n, int n_channels){
	dim3 blockSize(DIVUP(n, THREADS_PER_BLOCK));
	dim3 threadSize(THREADS_PER_BLOCK);
	grid2points_kernel<<<blockSize, threadSize>>>(points, indices, grid, n, n_channels);
}
