#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define M 4096
#define K 4096
#define N 4096



#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}


void initialize(float *a, int size){
    for(int i=0; i<size; i++){
        a[i] = i%1000;
    }
}

int main(){
    float *h_a = (float*) malloc(M * K * sizeof(float));
    float *h_b = (float*) malloc(K * N * sizeof(float));

    initialize(h_a,M*K);
    initialize(h_b,N*K);

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, M * K * sizeof(float));
    cudaMalloc(&d_b, N * K * sizeof(float));
    cudaMalloc(&d_c, M * N * sizeof(float));

    cudaMemcpy(d_a, h_a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K * N * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float alpha = 1.0f, beta = 0.0f;

    for(int i=0; i<10; i++){
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N));
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    double total = 0;
    int runs = 20;
    for(int i=0; i<runs; i++){
        cudaEventRecord(start);
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms,start, stop);    
        total += ms;    
    }
    
    double avg_ms = total / runs;
    double avg_s = avg_ms / 1000;

    double total_flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = total_flops / (avg_s * 1.0e9);
    double tflops = gflops/1000;
    
    printf(
        "M=%d N=%d K=%d | runs=%d\n"
        "Avg time = %.3f ms (%.6f s) | GFLOPS = %.2f | TFLOPs = %.2f\n",
        M, N, K, runs,
        avg_ms, avg_s, gflops, tflops
    );
    CHECK_CUBLAS(cublasDestroy(handle));

}

