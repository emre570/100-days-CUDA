#include <iostream>

__global__ void matrixAdd(float *mat1, float *mat2, float *sum_mat, int rows, int cols){
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // sütun indeksi
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // satır indeksi

    // Eleman indeksini hesaplayalım (row-major)
    int idx = row * cols + col;

    if (row < rows && col < cols){
        sum_mat[idx] = mat1[idx] + mat2[idx];
    }
}

void matAdd(float* mat1_h, float* mat2_h, float* sum_mat_h, int rows, int cols) {
    int size = (rows * cols) * sizeof(float);

    float *mat1_d, *mat2_d, *sum_mat_d;

    cudaMalloc((void**)&mat1_d, size);
    cudaMalloc((void**)&mat2_d, size);
    cudaMalloc((void**)&sum_mat_d, size);

    cudaMemcpy(mat1_d, mat1_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(mat2_d, mat2_h, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);  // ideal thread bloğu
    dim3 gridSize(
        (cols + blockSize.x - 1) / blockSize.x,
        (rows + blockSize.y - 1) / blockSize.y);

    // Start CUDA event timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
 
    cudaEventRecord(start);

    matrixAdd<<<gridSize, blockSize>>>(mat1_d, mat2_d, sum_mat_d, rows, cols);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Compute execution time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(sum_mat_h, sum_mat_d, size, cudaMemcpyDeviceToHost);

    cudaFree(mat1_d);
    cudaFree(mat2_d);
    cudaFree(sum_mat_d);

    std::cout << "Kernel Execution Time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(){
    size_t rows = 5000, cols = 5000;
    size_t mat_size = rows * cols;

    float *mat1 = new float[mat_size];
    float *mat2 = new float[mat_size];
    float *sum_mat = new float[mat_size];

    for (int i = 0; i < mat_size; i++) {
        mat1[i] = static_cast<float>(rand()) / RAND_MAX;
        mat2[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    matAdd(mat1, mat2, sum_mat, rows, cols);

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
    delete[] sum_mat;

    return 0;
}