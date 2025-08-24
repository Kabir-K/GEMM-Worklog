#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 4096  
#define K 4096 
#define N 4096
#define BLOCK_SIZE 32



__global__ void matmul_gpu(float *A, float *B, float *C, int m, int k, int n){

  const int cRow = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
  const int cCol = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);
  
  if (cRow < m && cCol < n) {
    float tmp = 0.0;
    for (int i = 0; i < k; ++i) {
      tmp += A[cRow * k + i] * B[i * n + cCol];
    }
    C[cRow * n + cCol] = tmp;
  }
}

void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}


double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    float *d_A, *d_B, *d_C;
    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C_cpu = (float*)malloc(size_C);
    h_C_gpu = (float*)malloc(size_C);

    srand(time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

 
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);


    for (int i = 0; i < 3; i++) {
        matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
    }

    double total = 0;
    int runs = 20;
    for (int i = 0; i < runs; i++) {
        double start_time = get_time();
        matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        total += end_time - start_time;
    }

    double avg_s = total / runs;
    double avg_ms = avg_s * 1000;

    double total_flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = total_flops / (avg_s * 1.0e9);
    double tflops = gflops/1000;
    
    printf(
        "M=%d N=%d K=%d | runs=%d\n"
        "Avg time = %.3f ms (%.6f s) | GFLOPS = %.2f | TFLOPs = %.2f\n",
        M, N, K, runs,
        avg_ms, avg_s, gflops, tflops
    );

    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
