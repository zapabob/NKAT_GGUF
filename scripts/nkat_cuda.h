
#ifndef NKAT_CUDA_H
#define NKAT_CUDA_H

#include "ggml.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// NKAT CUDA operations
void ggml_cuda_nkat_star_gemm(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * src2,
    struct ggml_tensor * dst,
    cudaStream_t stream
);

void ggml_cuda_nkat_benchmark(int M, int N, int K, int iterations);

#ifdef __cplusplus
}
#endif

#endif // NKAT_CUDA_H
