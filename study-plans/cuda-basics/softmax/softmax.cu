#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    __shared__ float sdata[256];

    int tid = threadIdx.x;

    float local_max = -1;

    for(int i = tid; i < N; i += blockDim.x) local_max = fmaxf(local_max, input[i]);

    sdata[tid] = local_max;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0;stride >>= 1){
        if(tid < stride)
            sdata[tid] =
                fmaxf(sdata[tid],
                      sdata[tid + stride]);

        __syncthreads();
    }

    float max_val = sdata[0];

    float local_sum = 0.0f;

    for(int i = tid; i < N; i += blockDim.x)
        local_sum += expf(input[i] - max_val);

    sdata[tid] = local_sum;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if(tid < stride)
            sdata[tid] += sdata[tid + stride];

        __syncthreads();
    }

    float sum_exp = sdata[0];

    for(int i = tid; i < N; i += blockDim.x){
        output[i] =
            expf(input[i] - max_val) / sum_exp;
    }
    
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}