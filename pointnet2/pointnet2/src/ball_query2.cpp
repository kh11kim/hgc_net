#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "ball_query2_gpu.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x, " must be a CUDA tensor ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

int ball_query2_wrapper_fast(int b, int n, int m, int nsample, at::Tensor radius_size_tensor,
    at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_tensor) {
    CHECK_INPUT(radius_size_tensor);
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(xyz_tensor);
    const float *radius_size = radius_size_tensor.data_ptr<float>();
    const float *new_xyz = new_xyz_tensor.data_ptr<float>();
    const float *xyz = xyz_tensor.data_ptr<float>();
    int *idx = idx_tensor.data_ptr<int>();
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    ball_query2_kernel_launcher_fast(b, n, m, nsample, radius_size ,new_xyz, xyz, idx, stream);
    return 1;
}
