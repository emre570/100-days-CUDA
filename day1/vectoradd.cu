#include <iostream>
#include <cuda_runtime.h>

__global__
void addVectorsKernel(float* vec1, float* vec2, float* sum_vec, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        sum_vec[i] = vec1[i] + vec2[i];
    }
}

void addVectors(float* vec1_h, float* vec2_h, float* sum_vec_h, int n) {
    int size = n * sizeof(float);
    float *vec1_d, *vec2_d, *sum_vec_d;

    cudaMalloc((void**)&vec1_d, size);
    cudaMalloc((void**)&vec2_d, size);
    cudaMalloc((void**)&sum_vec_d, size);

    cudaMemcpy(vec1_d, vec1_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(vec2_d, vec2_h, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    addVectorsKernel<<<numBlocks, threadsPerBlock>>>(vec1_d, vec2_d, sum_vec_d, n);

    cudaMemcpy(sum_vec_h, sum_vec_d, size, cudaMemcpyDeviceToHost);

    cudaFree(vec1_d);
    cudaFree(vec2_d);
    cudaFree(sum_vec_d);
}

int main() {
    const int vec_size = 100000;

    float *vec1 = new float[vec_size];
    float *vec2 = new float[vec_size];
    float *sum_vec = new float[vec_size];

    for (int i = 0; i < vec_size; i++) {
        vec1[i] = static_cast<float>(rand()) / RAND_MAX;
        vec2[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    addVectors(vec1, vec2, sum_vec, vec_size);

    std::cout << "First 10 results: ";
    for (int i = 0; i < 10; i++) {
        std::cout << sum_vec[i] << " ";
    }
    std::cout << std::endl;

    delete[] vec1;
    delete[] vec2;
    delete[] sum_vec;

    return 0;
}
