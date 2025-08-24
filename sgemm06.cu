#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define M 4096
#define N 4096
#define K 4096
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define alpha 1
#define beta 0



template <const int BM = 128, const int BN = 128, const int TM= 8, const int TN = 8, const int BK = 8>
__global__ void sgemm_6(float *A,float *B, float *C) {
const uint cRow = blockIdx.y;
const uint cCol = blockIdx.x;

const int threadCol = threadIdx.x % (BN / TN);
const int threadRow = threadIdx.x / (BN / TN);

__shared__ float As[BM * BK];
__shared__ float Bs[BK * BN];

A += cRow * BM * K;
B += cCol * BN;
C += cRow * BM * N + cCol * BN;


const uint innerRowA = threadIdx.x / (BK / 4);
const uint innerColA = threadIdx.x % (BK / 4);
const uint innerRowB = threadIdx.x / (BN / 4);
const uint innerColB = threadIdx.x % (BN / 4);


float threadResults[TM * TN] = {0.0};
float regM[TM] = {0.0};
float regN[TN] = {0.0};


for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    float4 tmp = reinterpret_cast<float4 *>(&A[innerRowA * K + threadCol * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
            reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    
    __syncthreads();

    A += BK;  
    B += BK * N;

    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
        for (uint i = 0; i < TM; ++i) {
            regM[i] = As[dotIdx * BM + threadRow * TM + i];
        }
        for(int i = 0; i < TN; i++){
            regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
        }
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
              threadResults[resIdxM * TN + resIdxN] +=
                  regM[resIdxM] * regN[resIdxN];
            }
        }
    }
    __syncthreads();
}

for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
        float4 tmp = reinterpret_cast<float4 *>(
        &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
        tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
        tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
        tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
        tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
        reinterpret_cast<float4 *>(
        &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
        tmp;
        }
    }
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

    int TM = 8;
    int TN = 8;
    int BM = 128;
    int BN = 128;


    dim3 threadsperblock((BM * BN) / (TM * TN));
    dim3 griddim(CEIL_DIV(N,BN),CEIL_DIV(M,BM));

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, M * K * sizeof(float));
    cudaMalloc(&d_b, N * K * sizeof(float));
    cudaMalloc(&d_c, M * N * sizeof(float));

    cudaMemcpy(d_a, h_a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K * N * sizeof(float), cudaMemcpyHostToDevice);

    for(int i=0; i<10; i++){
        sgemm_6<<<griddim,threadsperblock>>>(d_a, d_b, d_c);
        cudaDeviceSynchronize();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    double total = 0;
    int runs = 20;
    for(int i=0; i<runs; i++){
        cudaEventRecord(start);
        sgemm_6<<<griddim,threadsperblock>>>(d_a, d_b, d_c);
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


}
