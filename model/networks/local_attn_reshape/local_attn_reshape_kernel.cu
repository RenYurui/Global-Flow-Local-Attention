#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_NUM_THREADS 256 
#define CUDA_MAX_THREADS 256 

// #define THREADS_PER_BLOCK 64 

#define DIM0(TENSOR) ((TENSOR).x)
#define DIM1(TENSOR) ((TENSOR).y)
#define DIM2(TENSOR) ((TENSOR).z)
#define DIM3(TENSOR) ((TENSOR).w)

#define DIM3_INDEX(TENSOR, xx, yy, zz, ww) ((TENSOR)[((xx) * (TENSOR##_stride.x)) + ((yy) * (TENSOR##_stride.y)) + ((zz) * (TENSOR##_stride.z)) + ((ww) * (TENSOR##_stride.w))])
#define EPS 1e-8
#define SAFE_DIV(a, b)  ( (b==0)? ( (a)/(EPS) ): ( (a)/(b) )  )
#define CHECK_LEGALITY(x, min, max) ((x>=min && x<=max)? (true):(false) )

template <typename scalar_t>
__global__ void kernel_local_attn_reshape_update_output(const int n, 
                                                const scalar_t* __restrict__ inputs, 
                                                const long4 inputs_size, 
                                                const long4 inputs_stride,
                                                scalar_t* __restrict__ output, 
                                                const long4 output_size, 
                                                const long4 output_stride, 
                                                int kernel_size) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;


    if (index >= n) {
        return;
    }
    

    int dim_b   = DIM0(output_size);
    // int dim_c   = DIM1(output_size);
    int dim_h   = DIM2(output_size);
    int dim_w   = DIM3(output_size);
    int dim_chw = DIM0(output_stride);
    // int dim_hw  = DIM1(output_stride);



    int b = ( index / dim_chw ) % dim_b;
    // int c = ( index / dim_hw  ) % dim_c;
    int y = ( index / dim_w   ) % dim_h;
    int x = ( index           ) % dim_w;

    int ys = y/kernel_size;
    int xs = x/kernel_size;
    int yf_c = y%kernel_size;
    int xf_c = x%kernel_size;
    int cs = yf_c*kernel_size + xf_c;

    output[index] = DIM3_INDEX(inputs, b, cs, ys, xs);


}



template <typename scalar_t>
__global__ void kernel_local_attn_reshape_backward(
                                            const int n,  
                                            const scalar_t* __restrict__ inputs, 
                                            const long4 inputs_size, 
                                            const long4 inputs_stride,
                                            const scalar_t* __restrict__ grad_output, 
                                            const long4 grad_output_size, 
                                            const long4 grad_output_stride,
                                            scalar_t* __restrict__ grad_inputs, 
                                            const long4 grad_inputs_size, 
                                            const long4 grad_inputs_stride, 
                                            int kernel_size) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }


    int dim_b   = DIM0(grad_output_size);
    int dim_c   = DIM1(grad_output_size);
    int dim_h   = DIM2(grad_output_size);
    int dim_w   = DIM3(grad_output_size);
    int dim_chw = DIM0(grad_output_stride);
    int dim_hw  = DIM1(grad_output_stride);


    int b = ( index / dim_chw ) % dim_b;
    int c = ( index / dim_hw  ) % dim_c;
    int y = ( index / dim_w   ) % dim_h;
    int x = ( index           ) % dim_w;

    int ys = y/kernel_size;
    int xs = x/kernel_size;
    int yf_c = y%kernel_size;
    int xf_c = x%kernel_size;
    int cs = yf_c*kernel_size + xf_c;


    atomicAdd(&DIM3_INDEX(grad_inputs, b, cs, ys, xs), DIM3_INDEX(grad_output, b, c, y, x));

}

void local_attn_reshape_kernel_forward(
    at::Tensor& inputs,
    at::Tensor& output,
    int kernel_size) {
    // clock_t start, end;
    // start = clock();

    int n = output.numel();

    const long4 inputs_size = make_long4(inputs.size(0), inputs.size(1), inputs.size(2), inputs.size(3));
    const long4 inputs_stride = make_long4(inputs.stride(0), inputs.stride(1), inputs.stride(2), inputs.stride(3));

    const long4 output_size = make_long4(output.size(0), output.size(1), output.size(2), output.size(3));
    const long4 output_stride = make_long4(output.stride(0), output.stride(1), output.stride(2), output.stride(3));

    int Threads = CUDA_NUM_THREADS;

    const dim3 threads(Threads);
    const dim3 blocks((n + Threads - 1) / Threads);

    AT_DISPATCH_FLOATING_TYPES(inputs.type(), "local_attn_reshape_forward_kernel", ([&] {
        kernel_local_attn_reshape_update_output<scalar_t><<< blocks, threads, 0, at::cuda::getCurrentCUDAStream() >>>(
            n,
            inputs.data<scalar_t>(),
            inputs_size,
            inputs_stride, 
            output.data<scalar_t>(),
            output_size,
            output_stride,
            kernel_size);
    }));
    
    // end = clock();
    // printf("%d\n", end-start);
        // TODO: ATen-equivalent check

       //    THCudaCheck(cudaGetLastError());

}




void local_attn_reshape_kernel_backward(
    at::Tensor& inputs,
    at::Tensor& grad_output,
    at::Tensor& grad_inputs, 
    int kernel_size) {  

    int n = grad_output.numel();

    const long4 inputs_size = make_long4(inputs.size(0), inputs.size(1), inputs.size(2), inputs.size(3));
    const long4 inputs_stride = make_long4(inputs.stride(0), inputs.stride(1), inputs.stride(2), inputs.stride(3));

    const long4 grad_output_size = make_long4(grad_output.size(0), grad_output.size(1), grad_output.size(2), grad_output.size(3));
    const long4 grad_output_stride = make_long4(grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3));

    const long4 grad_inputs_size = make_long4(grad_inputs.size(0), grad_inputs.size(1), grad_inputs.size(2), grad_inputs.size(3));
    const long4 grad_inputs_stride = make_long4(grad_inputs.stride(0), grad_inputs.stride(1), grad_inputs.stride(2), grad_inputs.stride(3));


    int Threads = CUDA_NUM_THREADS;

    const dim3 threads(Threads);
    const dim3 blocks((n + Threads - 1) / Threads);

    AT_DISPATCH_FLOATING_TYPES(inputs.type(), "local_attn_reshape_backward", ([&] {
        kernel_local_attn_reshape_backward<scalar_t><<< blocks, threads, 0, at::cuda::getCurrentCUDAStream() >>>(
            n, 
            inputs.data<scalar_t>(), 
            inputs_size,
            inputs_stride,
            grad_output.data<scalar_t>(),
            grad_output_size,
            grad_output_stride,
            grad_inputs.data<scalar_t>(),
            grad_inputs_size,
            grad_inputs_stride, 
            kernel_size);

    }));
    // TODO: Use the ATen equivalent to get last error

    //    THCudaCheck(cudaGetLastError());

}