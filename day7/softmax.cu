#include <iostream>
#include <cuda_runtime.h>

__global__ void softmaxKernel(float* xd, float* resd, int M, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    float* input_row = xd + row * N;
    float* output_row = resd + row * N;

    float local_max = -INFINITY;
    float local_norm = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        float x = input_row[i];

        if (x > local_max) {
            local_norm *= expf(local_max - x);
            local_max = x;
        }

        local_norm += expf(x - local_max);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
        local_norm += __shfl_down_sync(0xffffffff, local_norm, offset);
    }

    __shared__ float smem_max[32]; 
    __shared__ float smem_norm[32];

    if (tid % 32 == 0) {
        smem_max[tid / 32] = local_max;
        smem_norm[tid / 32] = local_norm;
    }
    __syncthreads();

    if (blockDim.x > 32) {
        if (tid < 32) {
            local_max = smem_max[tid];
            local_norm = smem_norm[tid];

            for (int offset = 16; offset > 0; offset /= 2) {
                local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
                local_norm += __shfl_down_sync(0xffffffff, local_norm, offset);
            }
            if (tid == 0) {
                smem_max[0] = local_max;
                smem_norm[0] = local_norm;
            }
        }
    }
    __syncthreads();

    float row_max = smem_max[0];
    float row_norm = smem_norm[0];
    __syncthreads();

    for (int i = tid; i < N; i += blockDim.x) {
        output_row[i] = expf(input_row[i] - row_max) / row_norm;
    }
}