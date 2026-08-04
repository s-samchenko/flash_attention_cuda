#include "gpt2_gemm.hpp"

#include <cstdio>
#include <cstdlib>
#include <source_location>

namespace {

    void cublas_check(cublasStatus_t st, std::source_location loc = std::source_location::current()) {
        if (st != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "%s:%u cublas_check: status %d\n", loc.file_name(), loc.line(), int(st));
            std::exit(1);
        }
    }

}

namespace gpt2 {

    // Row-major C = A*B via the column-major identity C^T = B^T * A^T (operands swapped).
    void gemm_rm(cublasHandle_t handle, const float* A, const float* B, float* C, int m, int k, int n, float beta) {
        const float alpha = 1.f;
        cublas_check(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k, &beta, C, n));
    }

    // Row-major C = A*B^T, with B stored [n,k].
    void gemm_rm_bt(cublasHandle_t handle, const float* A, const float* B, float* C, int m, int k, int n) {
        const float alpha = 1.f, beta = 0.f;
        cublas_check(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, B, k, A, k, &beta, C, n));
    }

}
