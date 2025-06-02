
// Host interface for NKAT CUDA kernels
// Integration with ggml-cuda.cu

#include "ggml.h"
#include "ggml-cuda.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

// NKAT kernel declarations
extern "C" {
    void nkat_star_gemm_q8_v1(
        const int8_t* Aq, const int8_t* θq, const int8_t* xq,
        const float scaleA, const float scaleθ, const float scaleX,
        float* out, const int M, const int N, const int K
    );
    
    void nkat_star_gemm_q8_v2_optimized(
        const int8_t* Aq, const int8_t* θq, const int8_t* xq,
        const float scaleA, const float scaleθ, const float scaleX,
        float* out, const int M, const int N, const int K
    );
}

// Host function for NKAT star GEMM
void ggml_cuda_nkat_star_gemm(
    const ggml_tensor * src0,  // Weight matrix W
    const ggml_tensor * src1,  // Input vector/matrix x
    const ggml_tensor * src2,  // Theta matrix θ
    ggml_tensor * dst,         // Output
    cudaStream_t stream
) {
    GGML_ASSERT(src0->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(src1->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(src2->type == GGML_TYPE_Q8_0);
    
    const int M = src0->ne[1];  // Rows of weight matrix
    const int K = src0->ne[0];  // Cols of weight matrix / length of input
    const int N = src1->ne[1];  // Batch size or output width
    
    // Extract quantization scales
    const float scaleA = *(float*)((char*)src0->data + src0->nb[1] * M);
    const float scaleθ = *(float*)((char*)src2->data + src2->nb[1] * M);
    const float scaleX = *(float*)((char*)src1->data + src1->nb[0] * K);
    
    // Get device pointers
    const int8_t* d_Aq = (const int8_t*)src0->data;
    const int8_t* d_θq = (const int8_t*)src2->data;
    const int8_t* d_xq = (const int8_t*)src1->data;
    float* d_out = (float*)dst->data;
    
    // Configure kernel launch parameters
    const int TILE_SIZE = 32;
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    // Launch kernel based on matrix size
    if (M * N > 1024 * 1024) {
        // Large matrices: use memory-optimized version
        nkat_star_gemm_q8_memory_opt<<<gridSize, blockSize, 0, stream>>>(
            d_Aq, d_θq, d_xq, scaleA, scaleθ, scaleX, d_out, M, N, K
        );
    } else if (M * N > 64 * 64) {
        // Medium matrices: use warp-optimized version
        nkat_star_gemm_q8_v2_optimized<<<gridSize, blockSize, 0, stream>>>(
            d_Aq, d_θq, d_xq, scaleA, scaleθ, scaleX, d_out, M, N, K
        );
    } else {
        // Small matrices: use basic version
        nkat_star_gemm_q8_v1<<<gridSize, blockSize, 0, stream>>>(
            d_Aq, d_θq, d_xq, scaleA, scaleθ, scaleX, d_out, M, N, K
        );
    }
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "NKAT CUDA kernel error: %s\n", cudaGetErrorString(err));
    }
}

// Performance profiling function
void ggml_cuda_nkat_benchmark(int M, int N, int K, int iterations) {
    // Allocate test data
    int8_t *h_A, *h_θ, *h_x;
    float *h_out;
    
    h_A = (int8_t*)malloc(M * K * sizeof(int8_t));
    h_θ = (int8_t*)malloc(M * K * sizeof(int8_t));
    h_x = (int8_t*)malloc(K * N * sizeof(int8_t));
    h_out = (float*)malloc(M * N * sizeof(float));
    
    // Initialize with random data
    for (int i = 0; i < M * K; i++) {
        h_A[i] = rand() % 256 - 128;
        h_θ[i] = rand() % 256 - 128;
    }
    for (int i = 0; i < K * N; i++) {
        h_x[i] = rand() % 256 - 128;
    }
    
    // Allocate device memory
    int8_t *d_A, *d_θ, *d_x;
    float *d_out;
    
    cudaMalloc(&d_A, M * K * sizeof(int8_t));
    cudaMalloc(&d_θ, M * K * sizeof(int8_t));
    cudaMalloc(&d_x, K * N * sizeof(int8_t));
    cudaMalloc(&d_out, M * N * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_θ, h_θ, M * K * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, K * N * sizeof(int8_t), cudaMemcpyHostToDevice);
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const float scaleA = 0.01f, scaleθ = 0.005f, scaleX = 0.01f;
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        dim3 blockSize(32, 32);
        dim3 gridSize((N + 31) / 32, (M + 31) / 32);
        
        nkat_star_gemm_q8_v2_optimized<<<gridSize, blockSize>>>(
            d_A, d_θ, d_x, scaleA, scaleθ, scaleX, d_out, M, N, K
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    const float gflops = (2.0f * M * N * K * iterations) / (milliseconds * 1e6f);
    printf("NKAT CUDA Performance: %.2f GFLOPS (%.2f ms per iteration)\n", 
           gflops, milliseconds / iterations);
    
    // Cleanup
    free(h_A); free(h_θ); free(h_x); free(h_out);
    cudaFree(d_A); cudaFree(d_θ); cudaFree(d_x); cudaFree(d_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}
