#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_NUM_THREADS 256 
#define THREADS_PER_BLOCK 64 

#define DIM0(TENSOR) ((TENSOR).x)
#define DIM1(TENSOR) ((TENSOR).y)
#define DIM2(TENSOR) ((TENSOR).z)
#define DIM3(TENSOR) ((TENSOR).w)

#define DIM3_INDEX(TENSOR, xx, yy, zz, ww) ((TENSOR)[((xx) * (TENSOR##_stride.x)) + ((yy) * (TENSOR##_stride.y)) + ((zz) * (TENSOR##_stride.z)) + ((ww) * (TENSOR##_stride.w))])
#define EPS 1e-8
#define SAFE_DIV(a, b)  ( (b==0)? ( (a)/(EPS) ): ( (a)/(b) )  )




template <typename scalar_t>
__global__ void kernel_resample2d_update_output(const int n, 
                                               const scalar_t* __restrict__ input1, const long4 input1_size, const long4 input1_stride,
                                               const scalar_t* __restrict__ input2, const long4 input2_size, const long4 input2_stride, 
                                               scalar_t* __restrict__ output, const long4 output_size, const long4 output_stride, int kernel_size, int dilation) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    scalar_t val = 0.0f;
    scalar_t sum = 0.0f;


    int dim_b = DIM0(output_size);
    int dim_c = DIM1(output_size);
    int dim_h = DIM2(output_size);
    int dim_w = DIM3(output_size);
    int dim_chw = dim_c * dim_h * dim_w;
    int dim_hw  = dim_h * dim_w;

    int b = ( index / dim_chw ) % dim_b;
    int c = ( index / dim_hw )  % dim_c;
    int y = ( index / dim_w )   % dim_h;
    int x = ( index          )  % dim_w;

    scalar_t dx    = DIM3_INDEX(input2, b, 0, y, x);
    scalar_t dy    = DIM3_INDEX(input2, b, 1, y, x);
    scalar_t sigma = DIM3_INDEX(input2, b, 2, y, x);


    scalar_t xf = static_cast<scalar_t>(x) + dx;
    scalar_t yf = static_cast<scalar_t>(y) + dy;
    scalar_t alpha = xf - floor(xf); // alpha
    scalar_t beta = yf - floor(yf); // beta


    int idim_h = DIM2(input1_size);
    int idim_w = DIM3(input1_size);


    for (int fy = 0; fy < kernel_size/2; fy += 1) {
        int yT = max(min( int (floor(yf)-fy*dilation),    idim_h-1), 0);
        int yB = max(min( int (floor(yf)+(fy+1)*dilation),idim_h-1), 0);

        for (int fx = 0; fx < kernel_size/2; fx += 1) {
            int xL = max(min( int (floor(xf)-fx*dilation  ),    idim_w-1), 0);
            int xR = max(min( int (floor(xf)+(fx+1)*dilation),  idim_w-1), 0);

            scalar_t xL_ = ( static_cast<scalar_t>( fx    *dilation)+alpha );
            scalar_t xR_ = ( static_cast<scalar_t>((1.+fx)*dilation)-alpha );
            scalar_t yT_ = ( static_cast<scalar_t>( fy    *dilation)+beta  );
            scalar_t yB_ = ( static_cast<scalar_t>((1.+fy)*dilation)-beta  );

            scalar_t xL_P = exp(SAFE_DIV(-xL_*xL_, 2*sigma*sigma));
            scalar_t xR_P = exp(SAFE_DIV(-xR_*xR_, 2*sigma*sigma));
            scalar_t yT_P = exp(SAFE_DIV(-yT_*yT_, 2*sigma*sigma));
            scalar_t yB_P = exp(SAFE_DIV(-yB_*yB_, 2*sigma*sigma));
            // if (sigma==0){
            //     printf("xL_P %.10f\n", xL_P);
            //     // printf("%.10f\n", -(xL_*xL_)/(2*sigma*sigma));

            // }

            val += static_cast<scalar_t> (yT_P*xL_P * DIM3_INDEX(input1, b, c, yT, xL));
            val += static_cast<scalar_t> (yT_P*xR_P * DIM3_INDEX(input1, b, c, yT, xR));
            val += static_cast<scalar_t> (yB_P*xL_P * DIM3_INDEX(input1, b, c, yB, xL));
            val += static_cast<scalar_t> (yB_P*xR_P * DIM3_INDEX(input1, b, c, yB, xR));
            sum += (yT_P*xL_P + yT_P*xR_P + yB_P*xL_P + yB_P*xR_P);
        }
    }    

    output[index] = SAFE_DIV(val, sum);

}


template <typename scalar_t>
__global__ void kernel_resample2d_backward_input1(
    const int n, const scalar_t* __restrict__ input1, const long4 input1_size, const long4 input1_stride,
    const scalar_t* __restrict__ input2, const long4 input2_size, const long4 input2_stride,
    const scalar_t* __restrict__ gradOutput, const long4 gradOutput_size, const long4 gradOutput_stride,
    scalar_t* __restrict__ gradInput, const long4 gradInput_size, const long4 gradInput_stride, int kernel_size, int dilation) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    scalar_t sum = 0.0f;
    // scalar_t *xL_P = new scalar_t [kernel_size*kernel_size/4];
    // scalar_t *xR_P = new scalar_t [kernel_size*kernel_size/4];
    // scalar_t *yT_P = new scalar_t [kernel_size*kernel_size/4];
    // scalar_t *yB_P = new scalar_t [kernel_size*kernel_size/4];   

    int dim_b = DIM0(gradOutput_size);
    int dim_c = DIM1(gradOutput_size);
    int dim_h = DIM2(gradOutput_size);
    int dim_w = DIM3(gradOutput_size);
    int dim_chw = dim_c * dim_h * dim_w;
    int dim_hw  = dim_h * dim_w;

    int b = ( index / dim_chw ) % dim_b;
    int c = ( index / dim_hw )  % dim_c;
    int y = ( index / dim_w )   % dim_h;
    int x = ( index          )  % dim_w;

    scalar_t dx    = DIM3_INDEX(input2, b, 0, y, x);
    scalar_t dy    = DIM3_INDEX(input2, b, 1, y, x);
    scalar_t sigma = DIM3_INDEX(input2, b, 2, y, x);



    scalar_t xf = static_cast<scalar_t>(x) + dx;
    scalar_t yf = static_cast<scalar_t>(y) + dy;
    scalar_t alpha = xf - int(xf); // alpha
    scalar_t beta = yf - int(yf); // beta

    for (int fy = 0; fy < kernel_size/2; fy += 1) {
        for (int fx = 0; fx < kernel_size/2; fx += 1) {
            scalar_t xL_ = ( static_cast<scalar_t>( fx    *dilation)+alpha );
            scalar_t xR_ = ( static_cast<scalar_t>((1.+fx)*dilation)-alpha );
            scalar_t yT_ = ( static_cast<scalar_t>( fy    *dilation)+beta  );
            scalar_t yB_ = ( static_cast<scalar_t>((1.+fy)*dilation)-beta  );
            // scalar_t xL_ = ( alpha+static_cast<scalar_t>(fx) );
            // scalar_t xR_ = ( 1.-alpha+static_cast<scalar_t>(fx) );
            // scalar_t yT_ = ( beta+static_cast<scalar_t>(fy) );
            // scalar_t yB_ = ( 1-beta+static_cast<scalar_t>(fy) );

            scalar_t xL_P = exp(SAFE_DIV(-xL_*xL_, 2*sigma*sigma));
            scalar_t xR_P = exp(SAFE_DIV(-xR_*xR_, 2*sigma*sigma));
            scalar_t yT_P = exp(SAFE_DIV(-yT_*yT_, 2*sigma*sigma));
            scalar_t yB_P = exp(SAFE_DIV(-yB_*yB_, 2*sigma*sigma));
            // scalar_t xL_P = exp(SAFE_DIV(-xL_*xL_,2*sigma*sigma));
            // scalar_t xR_P = exp(-(xR_*xR_)/(2*sigma*sigma));
            // scalar_t yT_P = exp(-(yT_*yT_)/(2*sigma*sigma));
            // scalar_t yB_P = exp(-(yB_*yB_)/(2*sigma*sigma));          
            sum += (yT_P*xL_P + yT_P*xR_P + yB_P*xL_P + yB_P*xR_P);
            // printf("%f\n", SAFE_DIV(-xL_*xL_, 2*sigma*sigma));
        }
    }

    int idim_h = DIM2(input1_size);
    int idim_w = DIM3(input1_size);


    for (int fy = 0; fy < kernel_size/2; fy += 1) {
        int yT = max(min( int (floor(yf)-fy*dilation),    idim_h-1), 0);
        int yB = max(min( int (floor(yf)+(fy+1)*dilation),idim_h-1), 0);        
        // int yT = max(min( int (floor(yf)-fy  ),    idim_h-1), 0);
        // int yB = max(min( int (floor(yf)+fy+1),    idim_h-1), 0);

        for (int fx = 0; fx < kernel_size/2; fx += 1) {
            int xL = max(min( int (floor(xf)-fx*dilation  ),    idim_w-1), 0);
            int xR = max(min( int (floor(xf)+(fx+1)*dilation),  idim_w-1), 0);            
            // int xL = max(min( int (floor(xf)-fx  ),    idim_w-1), 0);
            // int xR = max(min( int (floor(xf)+fx+1),    idim_w-1), 0);

            scalar_t xL_ = ( static_cast<scalar_t>( fx    *dilation)+alpha );
            scalar_t xR_ = ( static_cast<scalar_t>((1.+fx)*dilation)-alpha );
            scalar_t yT_ = ( static_cast<scalar_t>( fy    *dilation)+beta  );
            scalar_t yB_ = ( static_cast<scalar_t>((1.+fy)*dilation)-beta  );
            // scalar_t xL_ = ( alpha+static_cast<scalar_t>(fx) );
            // scalar_t xR_ = ( 1.-alpha+static_cast<scalar_t>(fx) );
            // scalar_t yT_ = ( beta+static_cast<scalar_t>(fy) );
            // scalar_t yB_ = ( 1-beta+static_cast<scalar_t>(fy) );

            scalar_t xL_P = exp(SAFE_DIV(-xL_*xL_, 2*sigma*sigma));
            scalar_t xR_P = exp(SAFE_DIV(-xR_*xR_, 2*sigma*sigma));
            scalar_t yT_P = exp(SAFE_DIV(-yT_*yT_, 2*sigma*sigma));
            scalar_t yB_P = exp(SAFE_DIV(-yB_*yB_, 2*sigma*sigma));


            atomicAdd(&DIM3_INDEX(gradInput, b, c, (yT), (xL)), SAFE_DIV(yT_P*xL_P, sum) * DIM3_INDEX(gradOutput, b, c, y, x));
            atomicAdd(&DIM3_INDEX(gradInput, b, c, (yT), (xR)), SAFE_DIV(yT_P*xR_P, sum) * DIM3_INDEX(gradOutput, b, c, y, x));
            atomicAdd(&DIM3_INDEX(gradInput, b, c, (yB), (xL)), SAFE_DIV(yB_P*xL_P, sum) * DIM3_INDEX(gradOutput, b, c, y, x));
            atomicAdd(&DIM3_INDEX(gradInput, b, c, (yB), (xR)), SAFE_DIV(yB_P*xR_P, sum) * DIM3_INDEX(gradOutput, b, c, y, x));
        }
    }

}

template <typename scalar_t>
__global__ void kernel_resample2d_backward_input2(
    const int n, const scalar_t* __restrict__ input1, const long4 input1_size, const long4 input1_stride,
    const scalar_t* __restrict__ input2, const long4 input2_size, const long4 input2_stride,
    const scalar_t* __restrict__ gradOutput, const long4 gradOutput_size, const long4 gradOutput_stride,
    scalar_t* __restrict__ gradInput, const long4 gradInput_size, const long4 gradInput_stride, int kernel_size, int dilation) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    scalar_t grad1 = 0.0;
    scalar_t grad2 = 0.0;
    scalar_t sum = 0.0;

      

    int dim_b = DIM0(gradInput_size);
    int dim_c = DIM1(gradInput_size);
    int dim_h = DIM2(gradInput_size);
    int dim_w = DIM3(gradInput_size);
    int dim_chw = dim_c * dim_h * dim_w;
    int dim_hw  = dim_h * dim_w;

    int b = ( index / dim_chw ) % dim_b;
    int c = ( index / dim_hw )  % dim_c;
    int y = ( index / dim_w )   % dim_h;
    int x = ( index          )  % dim_w;

    int odim_c = DIM1(gradOutput_size);

    scalar_t dx    = DIM3_INDEX(input2, b, 0, y, x);
    scalar_t dy    = DIM3_INDEX(input2, b, 1, y, x);
    scalar_t sigma = DIM3_INDEX(input2, b, 2, y, x);


    scalar_t xf = static_cast<scalar_t>(x) + dx;
    scalar_t yf = static_cast<scalar_t>(y) + dy;
    scalar_t alpha = xf - floor(xf); // alpha
    scalar_t beta = yf - floor(yf); // beta


    int idim_h = DIM2(input1_size);
    int idim_w = DIM3(input1_size);
    scalar_t sumgrad = 0.0;

    for (int fy = 0; fy < kernel_size/2; fy += 1) {
        int yT = max(min( int (floor(yf)-fy*dilation),    idim_h-1), 0);
        int yB = max(min( int (floor(yf)+(fy+1)*dilation),idim_h-1), 0);  

        for (int fx = 0; fx < kernel_size/2; fx += 1) {
            int xL = max(min( int (floor(xf)-fx*dilation  ),    idim_w-1), 0);
            int xR = max(min( int (floor(xf)+(fx+1)*dilation),  idim_w-1), 0);  

            scalar_t xL_ = ( static_cast<scalar_t>( fx    *dilation)+alpha );
            scalar_t xR_ = ( static_cast<scalar_t>((1.+fx)*dilation)-alpha );
            scalar_t yT_ = ( static_cast<scalar_t>( fy    *dilation)+beta  );
            scalar_t yB_ = ( static_cast<scalar_t>((1.+fy)*dilation)-beta  );

            scalar_t xL_P = exp(SAFE_DIV(-xL_*xL_, 2*sigma*sigma));
            scalar_t xR_P = exp(SAFE_DIV(-xR_*xR_, 2*sigma*sigma));
            scalar_t yT_P = exp(SAFE_DIV(-yT_*yT_, 2*sigma*sigma));
            scalar_t yB_P = exp(SAFE_DIV(-yB_*yB_, 2*sigma*sigma));             
            sum += (yT_P*xL_P + yT_P*xR_P + yB_P*xL_P + yB_P*xR_P);

            for (int ch = 0; ch < odim_c; ++ch) {
                if (c==0) {
                    grad1 += SAFE_DIV(xL_ * yT_P * xL_P * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, yT, xL), -sigma*sigma);
                    grad1 -= SAFE_DIV(xR_ * yT_P * xR_P * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, yT, xR), -sigma*sigma);
                    grad1 += SAFE_DIV(xL_ * yB_P * xL_P * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, yB, xL), -sigma*sigma);
                    grad1 -= SAFE_DIV(xR_ * yB_P * xR_P * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, yB, xR), -sigma*sigma); 
                    sumgrad += SAFE_DIV((  xL_*yT_P*xL_P - xR_*yT_P*xR_P + xL_*yB_P*xL_P - xR_*yB_P*xR_P  ), -sigma*sigma);             
                }
                else if (c==1) {
                    grad1 += SAFE_DIV(yT_ * yT_P * xL_P * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, yT, xL), -sigma*sigma);
                    grad1 += SAFE_DIV(yT_ * yT_P * xR_P * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, yT, xR), -sigma*sigma);
                    grad1 -= SAFE_DIV(yB_ * yB_P * xL_P * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, yB, xL), -sigma*sigma);
                    grad1 -= SAFE_DIV(yB_ * yB_P * xR_P * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, yB, xR), -sigma*sigma);
                    sumgrad  += SAFE_DIV(( yT_*yT_P*xL_P + yT_*yT_P*xR_P - yB_*yB_P*xL_P - yB_*yB_P*xR_P  ), -sigma*sigma);             
                }
                else if (c==2) {
                    grad1 += SAFE_DIV((yT_*yT_+xL_*xL_) * yT_P * xL_P * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, yT, xL), sigma*sigma*sigma);                    
                    grad1 += SAFE_DIV((yT_*yT_+xR_*xR_) * yT_P * xR_P * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, yT, xR), sigma*sigma*sigma);
                    grad1 += SAFE_DIV((yB_*yB_+xL_*xL_) * yB_P * xL_P * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, yB, xL), sigma*sigma*sigma);
                    grad1 += SAFE_DIV((yB_*yB_+xR_*xR_) * yB_P * xR_P * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, yB, xR), sigma*sigma*sigma);
                    sumgrad  += SAFE_DIV((  (yT_*yT_+xL_*xL_)*yT_P*xL_P + (yT_*yT_+xR_*xR_)*yT_P*xR_P + (yB_*yB_+xL_*xL_)*yB_P*xL_P + (yB_*yB_+xR_*xR_)*yB_P*xR_P  ), sigma*sigma*sigma);  

                }
            }
        }
    }

        

    for (int fy = 0; fy < kernel_size/2; fy += 1) {
        int yT = max(min( int (floor(yf)-fy*dilation),    idim_h-1), 0);
        int yB = max(min( int (floor(yf)+(fy+1)*dilation),idim_h-1), 0);  

        for (int fx = 0; fx < kernel_size/2; fx += 1) {
            int xL = max(min( int (floor(xf)-fx*dilation  ),    idim_w-1), 0);
            int xR = max(min( int (floor(xf)+(fx+1)*dilation),  idim_w-1), 0);  

            scalar_t xL_ = ( static_cast<scalar_t>( fx    *dilation)+alpha );
            scalar_t xR_ = ( static_cast<scalar_t>((1.+fx)*dilation)-alpha );
            scalar_t yT_ = ( static_cast<scalar_t>( fy    *dilation)+beta  );
            scalar_t yB_ = ( static_cast<scalar_t>((1.+fy)*dilation)-beta  );

            scalar_t xL_P = exp(SAFE_DIV(-xL_*xL_, 2*sigma*sigma));
            scalar_t xR_P = exp(SAFE_DIV(-xR_*xR_, 2*sigma*sigma));
            scalar_t yT_P = exp(SAFE_DIV(-yT_*yT_, 2*sigma*sigma));
            scalar_t yB_P = exp(SAFE_DIV(-yB_*yB_, 2*sigma*sigma));
            for (int ch = 0; ch < odim_c; ++ch) {
                grad2 += sumgrad/odim_c * yT_P * xL_P * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, yT, xL);             
                grad2 += sumgrad/odim_c * yT_P * xR_P * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, yT, xR);                
                grad2 += sumgrad/odim_c * yB_P * xL_P * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, yB, xL);               
                grad2 += sumgrad/odim_c * yB_P * xR_P * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, yB, xR);   
            }
          
        }
    } 


    gradInput[index] = SAFE_DIV(grad1, sum) - SAFE_DIV(grad2, sum*sum);

}




void resample2d_kernel_forward(
    at::Tensor& input1, 
    at::Tensor& input2,
    at::Tensor& output, 
    int kernel_size,
    int dilation) {

    int n = output.numel();

    const long4 input1_size = make_long4(input1.size(0), input1.size(1), input1.size(2), input1.size(3));
    const long4 input1_stride = make_long4(input1.stride(0), input1.stride(1), input1.stride(2), input1.stride(3));

    const long4 input2_size = make_long4(input2.size(0), input2.size(1), input2.size(2), input2.size(3));
    const long4 input2_stride = make_long4(input2.stride(0), input2.stride(1), input2.stride(2), input2.stride(3));

    const long4 output_size = make_long4(output.size(0), output.size(1), output.size(2), output.size(3));
    const long4 output_stride = make_long4(output.stride(0), output.stride(1), output.stride(2), output.stride(3));

    // TODO: when atomicAdd gets resolved, change to AT_DISPATCH_FLOATING_TYPES_AND_HALF
    AT_DISPATCH_FLOATING_TYPES(input1.type(), "resample_forward_kernel", ([&] {
        kernel_resample2d_update_output<scalar_t><<< (n + CUDA_NUM_THREADS - 1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream() >>>(
            n,
            input1.data<scalar_t>(),
            input1_size,
            input1_stride, 
            input2.data<scalar_t>(),
            input2_size,
            input2_stride,
            output.data<scalar_t>(),
            output_size,
            output_stride,
            kernel_size,
            dilation);

    }));

        // TODO: ATen-equivalent check

       //    THCudaCheck(cudaGetLastError());

}

void resample2d_kernel_backward(
    at::Tensor& input1,
    at::Tensor& input2,
    at::Tensor& gradOutput,
    at::Tensor& gradInput1,
    at::Tensor& gradInput2,
    int kernel_size,
    int dilation) {

    int n = gradOutput.numel();

    const long4 input1_size = make_long4(input1.size(0), input1.size(1), input1.size(2), input1.size(3));
    const long4 input1_stride = make_long4(input1.stride(0), input1.stride(1), input1.stride(2), input1.stride(3));

    const long4 input2_size = make_long4(input2.size(0), input2.size(1), input2.size(2), input2.size(3));
    const long4 input2_stride = make_long4(input2.stride(0), input2.stride(1), input2.stride(2), input2.stride(3));

    const long4 gradOutput_size = make_long4(gradOutput.size(0), gradOutput.size(1), gradOutput.size(2), gradOutput.size(3));
    const long4 gradOutput_stride = make_long4(gradOutput.stride(0), gradOutput.stride(1), gradOutput.stride(2), gradOutput.stride(3));

    const long4 gradInput1_size = make_long4(gradInput1.size(0), gradInput1.size(1), gradInput1.size(2), gradInput1.size(3));
    const long4 gradInput1_stride = make_long4(gradInput1.stride(0), gradInput1.stride(1), gradInput1.stride(2), gradInput1.stride(3));

    AT_DISPATCH_FLOATING_TYPES(input1.type(), "resample_backward_input1", ([&] {

        kernel_resample2d_backward_input1<scalar_t><<< (n + CUDA_NUM_THREADS - 1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream() >>>(
            n, 
            input1.data<scalar_t>(), 
            input1_size,
            input1_stride,
            input2.data<scalar_t>(),
            input2_size, 
            input2_stride,
            gradOutput.data<scalar_t>(),
            gradOutput_size,
            gradOutput_stride,
            gradInput1.data<scalar_t>(),
            gradInput1_size,
            gradInput1_stride, 
            kernel_size,
            dilation
        );

    }));

    const long4 gradInput2_size = make_long4(gradInput2.size(0), gradInput2.size(1), gradInput2.size(2), gradInput2.size(3));
    const long4 gradInput2_stride = make_long4(gradInput2.stride(0), gradInput2.stride(1), gradInput2.stride(2), gradInput2.stride(3));

    n = gradInput2.numel();

    AT_DISPATCH_FLOATING_TYPES(gradInput2.type(), "resample_backward_input2", ([&] {


        kernel_resample2d_backward_input2<scalar_t><<< (n + CUDA_NUM_THREADS - 1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream() >>>(
            n, 
            input1.data<scalar_t>(), 
            input1_size, 
            input1_stride,
            input2.data<scalar_t>(), 
            input2_size,
            input2_stride,
            gradOutput.data<scalar_t>(),
            gradOutput_size,
            gradOutput_stride,
            gradInput2.data<scalar_t>(),
            gradInput2_size,
            gradInput2_stride,
            kernel_size,
            dilation
       );

    }));

    // TODO: Use the ATen equivalent to get last error

    //    THCudaCheck(cudaGetLastError());

}
