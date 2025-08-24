#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define M 4096
#define N 4096
#define K 4096
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define alpha 1
#define beta 0
int const WARPSIZE = 32;

template <const int BM = 128, const int BN = 128, const int BK = 16, const int WM = 64, const int WN = 64, const int WNITER = 4, const int TM = 8, const int TN = 4, const int NUM_THREADS = 128>
__global__ void __launch_bounds__(NUM_THREADS) sgemm_10(float *A, float *B, float *C){

    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    int warpId = threadIdx.x / WARPSIZE;

    int warpRow = warpId / (BN / WN);
    int warpCol = warpId % (BN / WN);

    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM  = WM / WMITER;
    constexpr uint WSUBN  = WN / WNITER;

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN + warpRow * WM * N + warpCol * WN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    

    int innerRowA = threadIdx.x / (BK / 4);
    int innerColA = threadIdx.x % (BK / 4);
    int strideA = ( NUM_THREADS * 4 ) / BK;
    int innerRowB = threadIdx.x / (BN / 4);
    int innerColB = threadIdx.x % (BN / 4);
    int strideB = ( NUM_THREADS * 4 ) / BN;

    int threadIdxInWarp = warpId % WARPSIZE;
    int threadColInWarp = threadIdxInWarp % (WSUBN / TN);
    int threadRowinWarp = threadIdxInWarp / (WSUBN / TN);

    float threadResults[TN * TM * WMITER * WNITER] = {0.0};
    float regM[TM * WMITER] = {0.0};
    float regN[TN * WNITER] = {0.0};

    for(int bk = 0; bk < K; bk += BK){

        for(int offset = 0; offset + strideA <= BM ; offset += strideA){

            float4 tmp = reinterpret_cast<float4 *>(&A[innerRowA * K + offset * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }

        for(int offset = 0; offset + strideB <= BK; offset += strideB){
            reinterpret_cast<float4 *>(&Bs[innerRowB * BN + offset * BN + innerColB * 4])[0] = 
                    reinterpret_cast<float4 *>(&B[innerRowB * N + offset * N + innerColB * 4])[0];
        }

        __syncthreads();

        A += BK;
        B += BK * N;

        for(int dotIdx = 0; dotIdx < BK; dotIdx++){
            for(int wSubRow = 0; wSubRow < WMITER; wSubRow++){
                for(int i=0; i < TM; i++){
                    regM[wSubRow * TM + i] = 
                            As[wSubRow * WSUBM + dotIdx * BM + i + warpRow * WM + threadRowinWarp * TM];
                }
            }
            for(int wSubCol = 0; wSubCol < WNITER; wSubCol++){
                for(int i=0; i < TN; i++){
                    regN[wSubCol * TN + i] = 
                            Bs[dotIdx * BN + threadColInWarp * TN + warpCol * WN + wSubCol * WSUBN + i];
                }
            }

            for(int wSubRow = 0; wSubRow < WMITER; wSubRow++){
                for(int wSubCol = 0; wSubCol < WNITER; wSubCol++){
                    for(int i = 0 ; i < TM; i++){
                        for(int j = 0; j < TN; j++){
                            threadResults[j + TN * i + TN * TM * wSubCol + TN * TM * WNITER * wSubRow] +=
                                regM[wSubRow * TM + i] * regN[wSubCol * TN + j];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    for(int wSubRow = 0; wSubRow < WMITER; wSubRow++){
        for(int wSubCol = 0; wSubCol < WNITER; wSubCol++){
            float *C_temp = C + wSubRow * WSUBM * N + wSubCol * WSUBN;
            for(int i = 0; i < TM; i++){
                for(int j = 0; j < TN; j += 4){
                    float4 tmp = 
                        reinterpret_cast<float4 *>(&C_temp[threadRowinWarp * TM * N + threadColInWarp * TN + i * N + j])[0];
                    
                        const int x = j + TN * i + TN * TM * wSubCol + TN * TM * WNITER * wSubRow;
                        
                        tmp.x = alpha * threadResults[x+0] + beta * tmp.x;
                        tmp.y = alpha * threadResults[x+1] + beta * tmp.y;
                        tmp.z = alpha * threadResults[x+2] + beta * tmp.z;
                        tmp.w = alpha * threadResults[x+3] + beta * tmp.w;
                        
                        reinterpret_cast<float4 *>(
                            &C_temp[threadRowinWarp * TM * N + threadColInWarp * TN + i * N + j])[0] = tmp;
                }
            }
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

    const int BM = 128;
    const int BN = 128;
    const int NUM_THREADS = 128;


    dim3 threadsperblock(NUM_THREADS);
    dim3 griddim(CEIL_DIV(N,BN),CEIL_DIV(M,BM));

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, M * K * sizeof(float));
    cudaMalloc(&d_b, N * K * sizeof(float));
    cudaMalloc(&d_c, M * N * sizeof(float));

    cudaMemcpy(d_a, h_a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K * N * sizeof(float), cudaMemcpyHostToDevice);

    for(int i=0; i<20; i++){
        sgemm_10<<<griddim,threadsperblock>>>(d_a, d_b, d_c);
        cudaDeviceSynchronize();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    double total = 0;
    int runs = 20;
    for(int i=0; i<runs; i++){
        cudaEventRecord(start);
        sgemm_10<<<griddim,threadsperblock>>>(d_a, d_b, d_c);
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
