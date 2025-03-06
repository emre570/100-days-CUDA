#include <iostream>

__global__ void matMul(float *mat1, float *mat2, float *mult_mat, int rows_a, int cols_a, int cols_b){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows_a && col < cols_b){
        float val = 0.0f;
        for (int k = 0; k < cols_a; ++k) {
            val += mat1[row * cols_a + k] * mat2[k * cols_b + col];
        }
        mult_mat[row * cols_b + col] = val;
    }
}

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

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (cols_b + blockSize.x - 1) / blockSize.x,
        (rows_a + blockSize.y - 1) / blockSize.y);

    // Start CUDA event timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
 
    cudaEventRecord(start);

    matMul<<<gridSize, blockSize>>>(mat1_d, mat2_d, mult_mat_d, rows_a, cols_a, cols_b);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Compute execution time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(mult_mat_h, mult_mat_d, sizeMult, cudaMemcpyDeviceToHost);

    cudaFree(mat1_d);
    cudaFree(mat2_d);
    cudaFree(mult_mat_d);

    std::cout << "Kernel Execution Time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(){
    size_t rows_a = 5, cols_a = 3, cols_b = 5;

    size_t size1 = rows_a * cols_a;
    size_t size2 = cols_a * cols_b;
    size_t sizeMult = rows_a * cols_b;

    float *mat1 = new float[size1];
    float *mat2 = new float[size2];
    float *mult_mat = new float[sizeMult];

    for (int i = 0; i < size1; i++) {
        mat1[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < size2; i++) {
        mat2[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    matMult(mat1, mat2, mult_mat, rows_a, cols_a, cols_b);

    //matAdd(mat1, mat2, sum_mat, rows, cols);

    // Print matrices
    // std::cout << "Matrix 1:\n";
    // for(int i=0; i<rows; i++){
    //     for(int j=0; j<cols; j++){
    //         std::cout << mat1[i*cols+j] << " ";
    //     }
    //     std::cout << "\n";
    // }

    // std::cout << "\nMatrix 2:\n";
    // for(int i=0; i<rows; i++){
    //     for(int j=0; j<cols; j++){
    //         std::cout << mat2[i*cols+j] << " ";
    //     }
    //     std::cout << "\n";
    // }

    // std::cout << "\nSum Matrix:\n";
    // for(int i=0; i<rows; i++){
    //     for(int j=0; j<cols; j++){
    //         std::cout << sum_mat[i*cols+j] << " ";
    //     }
    //     std::cout << "\n";
    // }

    delete[] mat1;
    delete[] mat2;
    delete[] mult_mat;

    return 0;
}