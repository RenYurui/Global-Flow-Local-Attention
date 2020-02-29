#include <ATen/ATen.h>
#include <torch/torch.h>

#include "block_extractor_kernel.cuh"
int block_extractor_cuda_forward(
    at::Tensor& source,
    at::Tensor& flow_field, 
    at::Tensor& output,
    int kernel_size) {
        block_extractor_kernel_forward(source, flow_field, output, kernel_size);
    return 1;
}

int block_extractor_cuda_backward(
    at::Tensor& source, 
    at::Tensor& flow_field,
    at::Tensor& grad_output,
    at::Tensor& grad_source, 
    at::Tensor& grad_flow_field, 
    int kernel_size) {
        block_extractor_kernel_backward(source, flow_field, grad_output,
                                 grad_source, grad_flow_field, 
                                 kernel_size);
    return 1;
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &block_extractor_cuda_forward, "BlockExtractor forward (CUDA)");
  m.def("backward", &block_extractor_cuda_backward, "BlockExtractor backward (CUDA)");
}

