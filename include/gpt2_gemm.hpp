#pragma once
#include <cublas_v2.h>

namespace gpt2 {

    // C[m,n] = A[m,k] * B[k,n]
    void gemm_rm(cublasHandle_t handle, const float* A, const float* B, float* C, int m, int k, int n, float beta = 0.f);

    // C[m,n] = A[m,k] * B[n,k]^T
    void gemm_rm_bt(cublasHandle_t handle, const float* A, const float* B, float* C, int m, int k, int n);

}
