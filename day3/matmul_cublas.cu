#include <iostream>
#include <cublas_v2.h>

void matMult(float *mat1_h, float *mat2_h, float *mult_mat_h, int rows_a, int cols_a, int cols_b){
    size_t size1 = (rows_a * cols_a) * sizeof(float);
    size_t size2 = (cols_a * cols_b) * sizeof(float);
    size_t sizeMult = (rows_a * cols_b) * sizeof(float);

    float *mat1_d, *mat2_d, *mult_mat_d;

    cudaMalloc((void**)&mat1_d, size1);
    cudaMalloc((void**)&mat2_d, size2);
    cudaMalloc((void**)&mult_mat_d, sizeMult);

    cudaMemcpy(mat1_d, mat1_h, size1, cudaMemcpyHostToDevice);
    cudaMemcpy(mat2_d, mat2_h, size2, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Start CUDA event timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
 
    cudaDeviceSynchronize();

    cudaEventRecord(start);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Dikkat: cuBLAS column-major düzen kullanır.
    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        cols_b, rows_a, cols_a,
        &alpha,
        mat2_d, cols_b,
        mat1_d, cols_a,
        &beta,
        mult_mat_d, cols_b);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Compute execution time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(mult_mat_h, mult_mat_d, sizeMult, cudaMemcpyDeviceToHost);

    std::cout << "Kernel Execution Time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(mat1_d);
    cudaFree(mat2_d);
    cudaFree(mult_mat_d);

    cublasDestroy(handle);
}

int main(){
    size_t rows_a = 4096, cols_a = 8192, cols_b = 4096;

    size_t size1 = rows_a * cols_a;
    size_t size2 = cols_a * cols_b;
    size_t sizeMult = rows_a * cols_b;

    float *mat1 = new float[size1];
    float *mat2 = new float[size2];
    float *mult_mat = new float[sizeMult];

    for (int i = 0; i < size1; i++)
        mat1[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < size2; i++)
        mat2[i] = static_cast<float>(rand()) / RAND_MAX;

    matMult(mat1, mat2, mult_mat, rows_a, cols_a, cols_b);

    // std::cout << "Multiplied Matrix:\n";
    // for(int i = 0; i < rows_a; i++){
    //     for(int j = 0; j < cols_b; j++)
    //         std::cout << mult_mat[i*cols_b+j] << " ";
    //     std::cout << "\n";
    // }

    delete[] mat1;
    delete[] mat2;
    delete[] mult_mat;

    return 0;
}
