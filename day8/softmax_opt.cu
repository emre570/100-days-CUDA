#include <iostream>
#include <curand.h>
#include <cuda_runtime.h>

#define ELEMENTS_PER_THREAD 16
#define MATRIX_ROWS 2048
#define MATRIX_COLS 2048
#define BLOCK_SIZE 512

void fill_curand(float *input, int size)
{
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    curandSetPseudoRandomGeneratorSeed(prng, 1234ULL);

    curandGenerateUniform(prng, input, size);
}

__global__ void softmaxKernel(float* input, float* output, int num_rows, int num_cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int base_idx = tid * ELEMENTS_PER_THREAD;

    float tile[ELEMENTS_PER_THREAD];
    
    float* input_row = input + row * num_cols;
    float* output_row = output + row * num_cols;
    
    float local_max = -INFINITY;
    float local_norm = 0.0f;

    #pragma unroll
    //Fill tile with ELEMENTS_PER_THREAD of elements
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        int idx = base_idx + i;
        if (idx < num_cols) {
            tile[i] = input_row[idx];
        }
    }

    #pragma unroll
    //Local max from tile
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        if (tile[i] > local_max)
            local_max = tile[i];
    }

    #pragma unroll
    //Local norm from tile
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        float x = tile[i];

        tile[i] = expf(x - local_max);
        local_norm += tile[i];
    }

    //Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
        local_norm += __shfl_down_sync(0xffffffff, local_norm, offset);
    }

    //Write SMEM per warp
    __shared__ float smem_max[32]; 
    __shared__ float smem_norm[32];

    if (tid % 32 == 0) {
        smem_max[tid / 32] = local_max;
        smem_norm[tid / 32] = local_norm;
    }
    __syncthreads();

    //Block-wise warp reduction
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

    //Final Softmax calculation
    //float row_max = smem_max[0];
    float row_norm = smem_norm[0];
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        int idx = base_idx + i;
        if (idx < num_cols) {
            output_row[idx] = tile[i] / row_norm;
        }
    }
}

void init_kernel(){
    size_t matrix_size = MATRIX_ROWS * MATRIX_COLS;
    float *h_input_matrix = new float[matrix_size];
    float *h_output_matrix = new float[matrix_size];

    float *d_input_matrix, *d_output_matrix;
    cudaMalloc((void**)&d_input_matrix, matrix_size * sizeof(float));
    cudaMalloc((void**)&d_output_matrix, matrix_size * sizeof(float));

    fill_curand(d_input_matrix, matrix_size);

    cudaMemcpy(d_input_matrix, h_input_matrix, matrix_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid_dim(MATRIX_ROWS);
    dim3 block_dim(BLOCK_SIZE);

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();

    cudaEventRecord(start);

    softmaxKernel<<<grid_dim, block_dim>>>(d_input_matrix, d_output_matrix, MATRIX_ROWS, MATRIX_COLS);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaEventRecord(stop);

    cudaDeviceSynchronize(); 

    cudaEventElapsedTime(&milliseconds, start, stop);

    uint64_t flops_per_element = 10;  // Yaklaşık ortalama
    uint64_t total_flop = matrix_size * flops_per_element;
    double tflops = (total_flop / (milliseconds / 1000.0)) / 1e12;

    std::cout << "Matrix Size(rows, cols): " << MATRIX_ROWS << ", " << MATRIX_COLS << std::endl;
    std::cout << "Elements Per Thread: " << ELEMENTS_PER_THREAD << std::endl;
    std::cout << "Block Size: " << BLOCK_SIZE << std::endl;
    std::cout << "Kernel Execution Time: " << milliseconds << " ms" << std::endl;
    std::cout << "TFLOP/s: " << tflops << " TFLOP/s" << std::endl;

    cudaMemcpy(h_output_matrix, d_output_matrix, matrix_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input_matrix);
    cudaFree(d_output_matrix);
}

int main(){
    init_kernel();
    return 0;
}