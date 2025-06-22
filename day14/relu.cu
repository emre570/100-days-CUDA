#include <vector>
#include <random>
#include <iostream>
#include <cuda_runtime.h>

std::vector<float> randomFill(std::vector<float> v) {
    // İstersen seed’i sabit tutarak reproducible yapabilirsin:
    static std::mt19937 gen(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto &x : v) {
        x = dist(gen);
    }
    return v;
}

void printVector(const std::vector<float>& vector){
    int len = static_cast<int>(vector.size());

    std::cout << "[";
    for(int i = 0; i < len; ++i){
        std::cout << vector[i] << (i+1< len ? ", " : "");
    }
    std::cout << "]\n";
}

__global__ void reluKernel(
    const float* __restrict__ A,
    float* B, int M, int N
){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N){
        int idx = row * N + col;
        float v = A[idx];
        B[idx] = v > 0 ? v : 0;
    }
}

void call_kernel(int M, int N, int block_size){
    int total_size = M * N;
    std::vector<float> A(M * N), B(M * N);
    
    A = randomFill(std::move(A));
    
    float *d_A, *d_B;
    cudaMalloc(&d_A, total_size * sizeof(float));
    cudaMalloc(&d_B, total_size * sizeof(float));

    cudaMemcpy(d_A, A.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int grid_x = (N + block_size - 1) / block_size;
    int grid_y = (M + block_size - 1) / block_size;
    
    dim3 grid(grid_x, grid_y);
    dim3 block(block_size, block_size);

    cudaEventRecord(start);

    reluKernel<<<grid, block>>>(d_A, d_B, M, N);

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(B.data(), d_B, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    double time_s = ms * 1e-3;
    double flops  = 2.0 * double(total_size);
    double gflops = flops / (time_s * 1e9);

    double bytes = 2.0 * double(total_size) * sizeof(float);
    double bandwidth = bytes / (time_s * 1e9);

    std::cout << M << "\t"<< N << "\t" 
    << block_size << "\t"
    << grid_x << "\t" << grid_y << "\t"
    << ms << "\t" << gflops << "\t" << bandwidth << "\n";

    cudaFree(d_A);
    cudaFree(d_B);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}

int main(){;
    std::vector<int> BLOCK_SIZES = {32, 64, 128};
    std::vector<int> A_x = {6144, 8192};
    std::vector<int> A_y = {6144, 8192};

    std::cout << "#M\t#N\t#BL\t#GRX\t#GRY\tTime(ms)\tGFLOPS\tBandwidth\n";

    for (int bl: BLOCK_SIZES){
        for (int m: A_x){
            for (int n: A_y){
                call_kernel(m, n, bl);
            }
        }
    }

    return 0;
}