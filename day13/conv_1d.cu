#include <vector>
#include <random>
#include <iostream>
#include <cuda_runtime.h>

#define MAX_K 8192 // constant memory size for kernel

// Constant memory for convolution kernel B
__constant__ float B_const[MAX_K];

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

std::vector<float> createPadding(
    const std::vector<float> A, 
    const std::vector<float> B,
    int N, int K,
    bool print_padded
){
    int pad_total = K - 1;
    int pad_left = pad_total / 2;

    int M = N + pad_total;

    std::vector<float> A_pad(M, 0.0f);

    for(int i = 0; i < N; ++i){
        A_pad[pad_left + i] = A[i];
    }

    if(print_padded){
        printVector(A_pad);
    }

    return A_pad;
}
__global__ void conv1d_kernel(
    const float* __restrict__ A_pad,
    //const float* __restrict__ B,
    float* C,
    int N,
    int K
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){
        float acc = 0; 

        //Convolution process
        #pragma unroll
        for(int j = 0; j < K; ++j){
            acc += A_pad[i + j] * B_const[j];
        }
        C[i] = acc;
    }
}

void call_kernel(int N, int K, int BLOCK_SIZE){
    //Create A and B vectors and apply dynamic padding
    std::vector<float> A(N), B(K);
    A = randomFill(std::move(A));
    B = randomFill(std::move(B));

    // Copy kernel B to constant memory
    cudaMemcpyToSymbol(B_const, B.data(), K * sizeof(float));

    bool print_padded = false;

    std::vector<float> A_pad = createPadding(A, B, N, K, print_padded);
    int M = static_cast<int>(A_pad.size());

    //Allocate memory for all vectors in GPU
    float *d_A_pad, *d_B, *d_C;
    cudaMalloc(&d_A_pad, (M * sizeof(float)));
    cudaMalloc(&d_B, (K * sizeof(float)));
    cudaMalloc(&d_C, (N * sizeof(float)));

    cudaMemcpy(d_A_pad, A_pad.data(), (M * sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), (K * sizeof(float)), cudaMemcpyHostToDevice);

    //Init event timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    //Call kernel
    cudaEventRecord(start);

    conv1d_kernel<<<gridSize, BLOCK_SIZE>>>(d_A_pad, d_C, N, K);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    //Get calculated C vector to host
    std::vector<float> C(N);
    cudaMemcpy(C.data(), d_C, (N * sizeof(float)), cudaMemcpyDeviceToHost);

    //printVector(C);

    //GFLOPS and time calculation
    double time_s = ms * 1e-3;
    double flops  = 2.0 * double(N) * double(K);
    double gflops = flops / (time_s * 1e9);

    // Print benchmark result
    std::cout << N << "\t"<< K << "\t" << BLOCK_SIZE 
              << "\t" << ms << "\t" << gflops << "\n";

    cudaFree(d_A_pad);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(){
    // Benchmark sizes for A and B
    std::vector<int> BLOCK_SIZES = {16, 32, 64, 128};
    std::vector<int> A_sizes = {65536, 32768, 131072, 524288};
    std::vector<int> B_sizes = {8191};

    std::cout << "#A\t#B\t#BLS\tTime(ms)\tGFLOPS\n";

    for(int N: A_sizes){
        for(int K: B_sizes){
            for(int BLK_SIZE: BLOCK_SIZES){
                call_kernel(N, K, BLK_SIZE);
            }
        }
    }
    
    return 0;
}