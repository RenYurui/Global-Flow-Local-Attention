#include <ATen/ATen.h>
#include <torch/torch.h>

#include "local_attn_reshape_kernel.cuh"
int local_attn_reshape_cuda_forward(
    at::Tensor& inputs,
    at::Tensor& output,
    int kernel_size) {
        local_attn_reshape_kernel_forward(inputs, output, kernel_size);
    return 1;
}

int local_attn_reshape_cuda_backward(
    at::Tensor& inputs, 
    at::Tensor& grad_output,
    at::Tensor& grad_inputs, 
    int kernel_size) {
        local_attn_reshape_kernel_backward(inputs, grad_output,
                                 grad_inputs, kernel_size);
    return 1;
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &local_attn_reshape_cuda_forward, "LocalAttnReshape forward (CUDA)");
  m.def("backward", &local_attn_reshape_cuda_backward, "LocalAttnReshape backward (CUDA)");
}

