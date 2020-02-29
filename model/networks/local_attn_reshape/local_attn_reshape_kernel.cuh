#pragma once

#include <ATen/ATen.h>

void local_attn_reshape_kernel_forward(
    at::Tensor& inputs,
    at::Tensor& output,
    int kernel_size);

void local_attn_reshape_kernel_backward(
    at::Tensor& inputs,
    at::Tensor& grad_output,
    at::Tensor& grad_inputs, 
    int kernel_size);