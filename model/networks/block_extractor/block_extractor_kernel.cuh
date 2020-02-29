#pragma once

#include <ATen/ATen.h>

void block_extractor_kernel_forward(
    at::Tensor& source,
    at::Tensor& flow_field,
    at::Tensor& output,
    int kernel_size);

void block_extractor_kernel_backward(
    at::Tensor& source,
    at::Tensor& flow_field,
    at::Tensor& grad_output,
    at::Tensor& grad_source, 
    at::Tensor& grad_flow_field, 
    int kernel_size);