#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cmath>

#include "C:\Users\radio\Documents\GitHub\100-days-CUDA\day9\include\matmul.cuh"
#include "C:\Users\radio\Documents\GitHub\100-days-CUDA\day9\include\transpose.cuh"
#include "C:\Users\radio\Documents\GitHub\100-days-CUDA\day9\include\softmax.cuh"

#define BATCH 3
#define HEADS 8
#define SEQ_LEN 4096
#define D_K 32

int main() {
    int B = BATCH, H = HEADS, S = SEQ_LEN, D = D_K;
    int total_heads = B * H;

    int qkv_size = total_heads * S * D;
    int attn_scores_size = total_heads * S * S;
    int output_size = total_heads * S * D;

    // CUDA bellekler
    float *Q, *K, *V, *K_T, *attn_scores, *softmax_output, *final_output;

    cudaMalloc(&Q, qkv_size * sizeof(float));
    cudaMalloc(&K, qkv_size * sizeof(float));
    cudaMalloc(&V, qkv_size * sizeof(float));
    cudaMalloc(&K_T, qkv_size * sizeof(float)); // transposed K
    cudaMalloc(&attn_scores, attn_scores_size * sizeof(float));
    cudaMalloc(&softmax_output, attn_scores_size * sizeof(float));
    cudaMalloc(&final_output, output_size * sizeof(float));

    // Doldur Q, K, V
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, 1234ULL);
    curandGenerateUniform(prng, Q, qkv_size);
    curandGenerateUniform(prng, K, qkv_size);
    curandGenerateUniform(prng, V, qkv_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float t_transpose = 0, t_qk_matmul = 0, t_softmax = 0, t_v_matmul = 0;

    cudaEventRecord(start);
    for (int i = 0; i < total_heads; i++) {
        launch_transpose(
            K + i * S * D,
            K_T + i * D * S,
            D, S
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_transpose, start, stop);
    std::cout << "Transpose time: " << t_transpose << " ms" << std::endl;

    cudaEventRecord(start);
    for (int i = 0; i < total_heads; i++) {
        launch_matmul(
            Q + i * S * D,
            K_T + i * D * S,
            attn_scores + i * S * S,
            S, S, D
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_qk_matmul, start, stop);
    std::cout << "Q × K^T time: " << t_qk_matmul << " ms" << std::endl;

    cudaEventRecord(start);
    launch_softmax(attn_scores, softmax_output, total_heads * S, S);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_softmax, start, stop);
    std::cout << "Softmax time: " << t_softmax << " ms" << std::endl;

    cudaEventRecord(start);
    for (int i = 0; i < total_heads; i++) {
        launch_matmul(
            softmax_output + i * S * S,
            V + i * S * D,
            final_output + i * S * D,
            S, D, S
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_v_matmul, start, stop);
    std::cout << "Softmax × V time: " << t_v_matmul << " ms" << std::endl;

    float total_time = t_transpose + t_qk_matmul + t_softmax + t_v_matmul;
    std::cout << "Total: " << total_time << " ms" << std::endl;

    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(K_T);
    cudaFree(attn_scores);
    cudaFree(softmax_output);
    cudaFree(final_output);

    return 0;
}
