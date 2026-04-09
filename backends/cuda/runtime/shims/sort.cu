/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <executorch/backends/aoti/utils.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/backends/cuda/runtime/shims/sort.h>
#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::backends::cuda {

namespace c10_slim = executorch::backends::aoti::slim::c10;

namespace {

__global__ void init_indices_kernel(
    int64_t* data,
    int64_t slice_size,
    int64_t total) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < total) {
    data[idx] = idx % slice_size;
  }
}

template <typename T>
void sort_slice_impl(
    T* keys,
    int64_t* values,
    int64_t n,
    bool descending,
    bool stable,
    cudaStream_t stream) {
  auto k = thrust::device_pointer_cast(keys);
  auto v = thrust::device_pointer_cast(values);
  if (stable && descending) {
    thrust::stable_sort_by_key(
        thrust::cuda::par.on(stream), k, k + n, v, thrust::greater<T>());
  } else if (stable) {
    thrust::stable_sort_by_key(
        thrust::cuda::par.on(stream), k, k + n, v);
  } else if (descending) {
    thrust::sort_by_key(
        thrust::cuda::par.on(stream), k, k + n, v, thrust::greater<T>());
  } else {
    thrust::sort_by_key(thrust::cuda::par.on(stream), k, k + n, v);
  }
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

AOTITorchError aoti_torch_cuda_sort_stable(
    Tensor* self,
    int32_t* stable,
    int64_t dim,
    int32_t descending,
    Tensor** ret0,
    Tensor** ret1) {
  ET_CHECK_OR_RETURN_ERROR(
      self != nullptr,
      InvalidArgument,
      "aoti_torch_cuda_sort_stable: self is null");
  ET_CHECK_OR_RETURN_ERROR(
      ret0 != nullptr,
      InvalidArgument,
      "aoti_torch_cuda_sort_stable: ret0 is null");
  ET_CHECK_OR_RETURN_ERROR(
      ret1 != nullptr,
      InvalidArgument,
      "aoti_torch_cuda_sort_stable: ret1 is null");

  int64_t ndim = static_cast<int64_t>(self->dim());

  if (dim < 0)
    dim += ndim;
  ET_CHECK_OR_RETURN_ERROR(
      dim >= 0 && dim < ndim,
      InvalidArgument,
      "aoti_torch_cuda_sort_stable: dim out of range");

  ET_CHECK_OR_RETURN_ERROR(
      self->is_contiguous(),
      NotSupported,
      "aoti_torch_cuda_sort_stable: non-contiguous input not supported");

  int64_t sort_size = self->size(dim);
  int64_t total_elements = static_cast<int64_t>(self->numel());
  int64_t num_slices = (sort_size > 0) ? total_elements / sort_size : 0;

  auto stream_result = getCurrentCUDAStream(0);
  ET_CHECK_OR_RETURN_ERROR(
      stream_result.ok(),
      Internal,
      "aoti_torch_cuda_sort_stable: failed to get CUDA stream");
  cudaStream_t stream = stream_result.get();

  // Contiguous strides for output tensors
  auto input_sizes = self->sizes();
  std::vector<int64_t> contig_strides(ndim);
  if (ndim > 0) {
    contig_strides[ndim - 1] = 1;
    for (int64_t i = ndim - 2; i >= 0; --i) {
      contig_strides[i] = contig_strides[i + 1] * input_sizes[i + 1];
    }
  }

  int32_t dtype_val = static_cast<int32_t>(self->dtype());

  // Allocate output values (same shape/dtype as input)
  *ret0 = nullptr;
  aoti_torch_empty_strided(
      ndim,
      input_sizes.data(),
      contig_strides.data(),
      dtype_val,
      static_cast<int32_t>(c10_slim::DeviceType::CUDA),
      0,
      ret0);
  ET_CHECK_OR_RETURN_ERROR(
      *ret0 != nullptr,
      Internal,
      "aoti_torch_cuda_sort_stable: failed to allocate values tensor");

  // Copy input data to output values
  if (total_elements > 0) {
    ET_CUDA_CHECK_OR_RETURN_ERROR(cudaMemcpyAsync(
        (*ret0)->data_ptr(),
        self->data_ptr(),
        self->nbytes(),
        cudaMemcpyDeviceToDevice,
        stream));
  }

  // Allocate output indices (same shape, int64 dtype)
  *ret1 = nullptr;
  aoti_torch_empty_strided(
      ndim,
      input_sizes.data(),
      contig_strides.data(),
      static_cast<int32_t>(c10_slim::ScalarType::Long),
      static_cast<int32_t>(c10_slim::DeviceType::CUDA),
      0,
      ret1);
  ET_CHECK_OR_RETURN_ERROR(
      *ret1 != nullptr,
      Internal,
      "aoti_torch_cuda_sort_stable: failed to allocate indices tensor");

  // Initialize indices: each slice gets 0, 1, ..., sort_size-1
  if (total_elements > 0) {
    int threads = 256;
    int blocks = static_cast<int>((total_elements + threads - 1) / threads);
    init_indices_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<int64_t*>((*ret1)->data_ptr()),
        sort_size,
        total_elements);
    ET_CUDA_KERNEL_LAUNCH_CHECK_OR_RETURN_ERROR();
  }

  if (sort_size <= 1) {
    return Error::Ok;
  }

  bool is_stable = (stable != nullptr && *stable != 0);
  bool desc = (descending != 0);

  // Require sorting along a contiguous dimension (stride == 1)
  int64_t dim_stride = self->stride(dim);
  ET_CHECK_OR_RETURN_ERROR(
      dim_stride == 1 || ndim == 1,
      NotSupported,
      "aoti_torch_cuda_sort_stable: sort along non-innermost dim "
      "of multi-D tensor not yet supported");

  auto self_dtype = self->dtype();

  for (int64_t s = 0; s < num_slices; ++s) {
    int64_t offset = s * sort_size;
    int64_t* idx_ptr =
        static_cast<int64_t*>((*ret1)->data_ptr()) + offset;

    switch (self_dtype) {
      case c10_slim::ScalarType::Long: {
        sort_slice_impl(
            static_cast<int64_t*>((*ret0)->data_ptr()) + offset,
            idx_ptr,
            sort_size,
            desc,
            is_stable,
            stream);
        break;
      }
      case c10_slim::ScalarType::Int: {
        sort_slice_impl(
            static_cast<int32_t*>((*ret0)->data_ptr()) + offset,
            idx_ptr,
            sort_size,
            desc,
            is_stable,
            stream);
        break;
      }
      case c10_slim::ScalarType::Float: {
        sort_slice_impl(
            static_cast<float*>((*ret0)->data_ptr()) + offset,
            idx_ptr,
            sort_size,
            desc,
            is_stable,
            stream);
        break;
      }
      default:
        ET_LOG(
            Error,
            "aoti_torch_cuda_sort_stable: unsupported dtype %d",
            static_cast<int>(self_dtype));
        return Error::InvalidArgument;
    }
  }

  ET_CUDA_KERNEL_LAUNCH_CHECK_OR_RETURN_ERROR();
  return Error::Ok;
}

#ifdef __cplusplus
}
#endif

} // namespace executorch::backends::cuda
