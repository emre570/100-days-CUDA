#include <cmath>
#include <vector>
#include <random>
#include <iostream>
#include <cuda_runtime.h>

std::vector<float> randomFill(std::vector<float> v) {
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

std::vector<float> applyPadding(
    const std::vector<float>& v,
    int pad_size
) {
    std::vector<float> padded;
    padded.reserve(v.size() + 2 * pad_size);

    padded.insert(padded.end(), pad_size, 0.0f);

    padded.insert(padded.end(), v.begin(), v.end());

    padded.insert(padded.end(), pad_size, 0.0f);

    return padded;
}

__global__ void avg_pool_1d_kernel(
    const float* __restrict__ input, int input_len,
    float* output, int out_len,
    int kernel_size, int stride
){
    extern __shared__ float smem[];
    int tid = threadIdx.x, bid = blockIdx.x;
    int block_offset = bid * blockDim.x * stride;
    int global_i = block_offset + tid * stride;

    // Load data to smem
    for (int i = tid; i < blockDim.x * stride + kernel_size; i += blockDim.x){
        int input_idx = block_offset + i;
        smem[i] = (input_idx < input_len) ? input[input_idx] : 0.0f;
    }
    __syncthreads();

    if (global_i + kernel_size <= input_len){
        float sum = 0.0f;
        int local_start = tid * stride;

        for (int k = 0; k < kernel_size; ++k){
            sum += smem[local_start + k];
        }

        output[global_i / stride] = sum / kernel_size;
    }
}

void callKernel(int blockSize, int Lsize, int Psize, int kSize, int sSize){
    // L for length of input, S for stride, P for padding, k for kernel length
    int L = Lsize, S = sSize, P = Psize, k = kSize;
    // Padded vector size
    int L_pad = L + 2 * P;
    // Output size
    int N_out = ceil(((L_pad - k) / S)) + 1;

    
    std::vector<float> input(L), output(N_out);
    input = randomFill(input);
    
    auto input_padded = applyPadding(input, P);
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, L * sizeof(float));
    cudaMalloc(&d_output, N_out * sizeof(float));
    
    cudaMemcpy(d_input, input_padded.data(), L * sizeof(float), cudaMemcpyHostToDevice);
    
    const int block_size = blockSize;
    int grid_size = ceil(N_out + block_size - 1) / block_size;
    int shared_bytes = (blockSize * S + k) * sizeof(float);
    
    dim3 block_dim(block_size);
    dim3 grid_dim(grid_size);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    avg_pool_1d_kernel<<<grid_dim, block_dim, shared_bytes>>>(
        d_input, L, d_output, N_out, k, S
    );
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaMemcpy(output.data(), d_output, N_out * sizeof(float), cudaMemcpyDeviceToHost);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << blockSize << "\t" << L << "\t" << P << "\t" << k << "\t" << S << "\t" << ms << "\n";

    cudaFree(d_input);
    cudaFree(d_output);
}

int main(){
    std::vector<int> BLOCK_SIZES = {128, 256, 512};
    std::vector<int> L_sizes = {32768, 262144};
    std::vector<int> P_sizes = {2048, 8192};
    std::vector<int> k_sizes = {128, 1024, 8192};
    std::vector<int> S_sizes = {1, 8, 128};
    
    std::cout << "#BLK\t#L\t#P\t#k\t#S\t#ms\n";
    
    for (int block: BLOCK_SIZES){
        for(int L: L_sizes){
            for(int P: P_sizes){
                for(int k: k_sizes){
                    for(int S: S_sizes){
                        callKernel(block, L, P, k, S);
                    }
                }
            }
        }
    }

    return 0;
}