#include <iostream>
#include <cuda_runtime.h>

#define TILE_WIDTH 32
#define MATRIX_ROWS 5120
#define MATRIX_COLS 6144

__global__ void tiled_transpose(float* original_matrix, float* transposed_matrix, int width, int height){
    __shared__ float tile[TILE_WIDTH][TILE_WIDTH + 1]; // +1 bank conflict azaltır

    int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int y = blockIdx.y * TILE_WIDTH + threadIdx.y;

    // Adım 1: Global'den Shared'a yükle
    if(x < width && y < height)
        tile[threadIdx.y][threadIdx.x] = original_matrix[y * width + x];

    __syncthreads();

    // Transpose edilmiş indeksi hesapla (x ve y yer değiştirir!)
    x = blockIdx.y * TILE_WIDTH + threadIdx.x; // dikkat burası önemli!
    y = blockIdx.x * TILE_WIDTH + threadIdx.y;

    if(x < height && y < width)
        transposed_matrix[y * height + x] = tile[threadIdx.x][threadIdx.y];
}
void host_transpose(){
    size_t matrix_size = MATRIX_ROWS * MATRIX_COLS;
    float *h_matrix = new float[matrix_size];
    float *h_t_matrix = new float[matrix_size];

    //Fill host matrix random numbers
    for(int i = 0; i < matrix_size; i++) {
        h_matrix[i] = (float)(rand() % 10);
    }

    //Write both matrices
    // Original matrix (2x3)
    // std::cout << "Orijinal Matris:" << std::endl;
    // for(int i = 0; i < MATRIX_ROWS; i++){
    //     for(int j = 0; j < MATRIX_COLS; j++){
    //         std::cout << h_matrix[i * MATRIX_COLS + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    float *d_matrix, *d_t_matrix;

    cudaMalloc((void**)&d_matrix, matrix_size * sizeof(float));
    cudaMemcpy(d_matrix, h_matrix, matrix_size * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_t_matrix, matrix_size * sizeof(float));  // BU EKSİK!


    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((MATRIX_COLS + TILE_WIDTH - 1)/TILE_WIDTH, 
                (MATRIX_ROWS + TILE_WIDTH - 1)/TILE_WIDTH);

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();  // Tüm önceki işlemler bitsin

    cudaEventRecord(start);

    tiled_transpose<<<gridDim, blockDim>>>(d_matrix, d_t_matrix, MATRIX_COLS, MATRIX_ROWS);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaEventRecord(stop);

    cudaDeviceSynchronize();  // Kernel tamamlanmadan zaman ölçümünü bitirme!

    cudaEventElapsedTime(&milliseconds, start, stop);

    uint64_t flops = 2ULL * matrix_size;
    double tflops = (flops / (milliseconds / 1000.0)) / 1e12;

    std::cout << "Matrix Size(rows, cols): " << MATRIX_ROWS << ", " << MATRIX_COLS << std::endl;
    std::cout << "Tile Width: " << TILE_WIDTH << std::endl;
    std::cout << "Kernel Execution Time: " << milliseconds << " ms" << std::endl;
    std::cout << "TFLOP/s: " << tflops << " TFLOP/s" << std::endl;

    cudaMemcpy(h_t_matrix, d_t_matrix, matrix_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_matrix);
    cudaFree(d_t_matrix);

    // Transpose matrix (3x2)
    // std::cout << "\nTranspose Matris:" << std::endl;
    // for(int i = 0; i < MATRIX_COLS; i++){
    //     for(int j = 0; j < MATRIX_ROWS; j++){
    //         std::cout << h_t_matrix[i * MATRIX_ROWS + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    //Clear matrices
    delete[] h_matrix;
    delete[] h_t_matrix;
}

int main(){
    host_transpose();
    return 0;
}