#pragma once

namespace gpt2 {

    // in-place elementwise gelu_new using tanh approximation
    void gelu_new(float* x, int n);

    // x[i] += bias[i % cols], bias broadcast down the rows
    void bias_add(float* x, const float* bias, int rows, int cols);

    // x[i] += y[i].
    void residual_add(float* x, const float* y, int n);

}

int gpt2_test_kernels(const char* fixtures_dir);
