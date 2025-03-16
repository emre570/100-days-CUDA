#include <iostream>
#include <cuda_runtime.h>

#define TILE_WIDTH 64
#define BLOCK_X 32
#define BLOCK_Y 32

__global__ void tiled_matmul(float* M, float* N, float* P, int Width){
    //Mds and Nds represents tiled pieces of M and N matrices
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    //Identify row and column of the P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    //Loop over the M and N tiles required to compute P element
    float PValue = 0;
    for(int ph = 0; ph < Width/TILE_WIDTH; ++ph){
        Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty)*Width + Col];

        __syncthreads();

        for(int k = 0; k < TILE_WIDTH; ++k){
            PValue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    P[Row * Width + Col] = PValue;
}
void call_matmul(){
    size_t m = 64, n = 24576, p = 1536;

    size_t size1 = m * p;
    size_t size2 = p * n;
    size_t sizeMult = m * n;

    float *h_M = new float[size1];
    float *h_N = new float[size2];
    float *h_P = new float[sizeMult];

    for(int i = 0; i < size1; i++) {
        h_M[i] = (float)(rand() % 10);
    }
    for(int i = 0; i < size2; i++) {
        h_N[i] = (float)(rand() % 10);
    }

    float *d_M, *d_N, *d_P;

    cudaMalloc((void**)&d_M, size1 * sizeof(float));
    cudaMalloc((void**)&d_N, size2 * sizeof(float));
    cudaMalloc((void**)&d_P, sizeMult * sizeof(float));

    cudaMemcpy(d_M, h_M, size1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size2 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_X, BLOCK_Y);
    dim3 gridSize(
        (n + TILE_WIDTH - 1) / TILE_WIDTH,
        (m + TILE_WIDTH - 1) / TILE_WIDTH);

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();  // Tüm önceki işlemler bitsin

    cudaEventRecord(start);

    tiled_matmul<<<gridSize, blockSize>>>(d_M, d_N, d_P, p);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaEventRecord(stop);

    cudaDeviceSynchronize();  // Kernel tamamlanmadan zaman ölçümünü bitirme!

    cudaEventElapsedTime(&milliseconds, start, stop);

    uint64_t flops = 2ULL * m * n * p;
    double tflops = (flops / (milliseconds / 1000.0)) / 1e12;

    std::cout << "Matrix Sizes(m,n,p): " << m << ", " << n << ", " << p << std::endl;
    std::cout << "Tile Width: " << TILE_WIDTH << std::endl;
    std::cout << "Block Size: " << BLOCK_X << ", " << BLOCK_Y << std::endl;
    std::cout << "Kernel Execution Time: " << milliseconds << " ms" << std::endl;
    std::cout << "TFLOP/s: " << tflops << " TFLOP/s" << std::endl;

    cudaMemcpy(h_P, d_P, sizeMult, cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    delete[] h_M;
    delete[] h_N;
    delete[] h_P;
}

int main(){
    call_matmul();
    return 0;
}