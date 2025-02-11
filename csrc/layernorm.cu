#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "dispatch_utils.h"
#ifndef USE_ROCM
  #include <cuda_bf16.h>
  #include <cuda_fp16.h>
  #include <cub/util_type.cuh>
  #include <cub/cub.cuh>
#else
  #include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>
  #include <hipcub/util_type.hpp>
  #include <hipcub/hipcub.hpp>

using __nv_bfloat16 = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;
#endif

namespace kizuna {

// TODO: Further optimize this kernel.
template <typename scalar_t>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ out,           // [..., hidden_size]
    const scalar_t* __restrict__ input,   // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float)input[blockIdx.x * hidden_size + idx];
    variance += x * x;
  }
  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)input[blockIdx.x * hidden_size + idx];
    out[blockIdx.x * hidden_size + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

/* Converter structs for the conversion from torch types to HIP/CUDA types,
   and the associated type conversions within HIP/CUDA. These helpers need
   to be implemented for now because the relevant type conversion
   operators/constructors are not consistently implemented by HIP/CUDA, so
   a generic conversion via type casts cannot be implemented.
   Each struct should have the member static constexpr bool `exists`:
   If false, the optimized kernel is not used for the corresponding torch type.
   If true, the struct should be fully defined as shown in the examples below.
 */
template <typename torch_type>
struct _typeConvert {
  static constexpr bool exists = false;
};

#if defined(USE_ROCM) || (defined(CUDA_VERSION) && (CUDA_VERSION >= 12000))
// CUDA < 12.0 runs into issues with packed type conversion
template <>
struct _typeConvert<c10::Half> {
  static constexpr bool exists = true;
  using hip_type = __half;
  using packed_hip_type = __half2;

  __device__ static inline float convert(hip_type x) { return __half2float(x); }
  __device__ static inline float2 convert(packed_hip_type x) {
    return __half22float2(x);
  }
  __device__ static inline hip_type convert(float x) {
    return __float2half_rn(x);
  }
  __device__ static inline packed_hip_type convert(float2 x) {
    return __float22half2_rn(x);
  }
};

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
// BFloat16 is only supported on Ampere (SM80) and newer
template <>
struct _typeConvert<c10::BFloat16> {
  static constexpr bool exists = true;
  using hip_type = __nv_bfloat16;
  using packed_hip_type = __nv_bfloat162;

  __device__ static inline float convert(hip_type x) {
    return __bfloat162float(x);
  }
  __device__ static inline float2 convert(packed_hip_type x) {
    return __bfloat1622float2(x);
  }
  __device__ static inline hip_type convert(float x) {
    return __float2bfloat16(x);
  }
  __device__ static inline packed_hip_type convert(float2 x) {
    return __float22bfloat162_rn(x);
  }
};
#else
// For pre-Ampere architectures, mark BF16 as unsupported
template <>
struct _typeConvert<c10::BFloat16> {
  static constexpr bool exists = false;
};
#endif  // __CUDA_ARCH__ >= 800
#else
// If __CUDA_ARCH__ is not defined, we're in host code
template <>
struct _typeConvert<c10::BFloat16> {
  static constexpr bool exists = true;  // Allow compilation but runtime will use fallback
};
#endif  // __CUDA_ARCH__
#endif  // defined(USE_ROCM) || ...

/* Vector POD struct to generate vectorized and packed FP16/BF16 ops
   for appropriate specializations of fused_add_rms_norm_kernel.
   Only functions that are necessary in that kernel are implemented.
   Alignment to 16 bytes is required to use 128-bit global memory ops.
 */
template <typename scalar_t, int width>
struct alignas(16) _f16Vec {
  /* Not theoretically necessary that width is a power of 2 but should
     almost always be the case for optimization purposes */
  static_assert(width > 0 && (width & (width - 1)) == 0,
                "Width is not a positive power of 2!");
  using Converter = _typeConvert<scalar_t>;
  using T1 = typename Converter::hip_type;
  using T2 = typename Converter::packed_hip_type;
  T1 data[width];

  __device__ _f16Vec& operator+=(const _f16Vec<scalar_t, width>& other) {
    if constexpr (width % 2 == 0) {
      #pragma unroll
      for (int i = 0; i < width; i += 2) {
        // Convert to float2 for higher precision addition
        float2 a = Converter::convert(T2{data[i], data[i + 1]});
        float2 b = Converter::convert(T2{other.data[i], other.data[i + 1]});
        a.x += b.x;
        a.y += b.y;
        T2 temp = Converter::convert(a);
        data[i] = temp.x;
        data[i + 1] = temp.y;
      }
    } else {
      #pragma unroll
      for (int i = 0; i < width; ++i) {
        float a = Converter::convert(data[i]);
        float b = Converter::convert(other.data[i]);
        data[i] = Converter::convert(a + b);
      }
    }
    return *this;
  }

  __device__ _f16Vec& operator*=(const _f16Vec<scalar_t, width>& other) {
    if constexpr (width % 2 == 0) {
      #pragma unroll
      for (int i = 0; i < width; i += 2) {
        // Convert to float2 for higher precision multiplication
        float2 a = Converter::convert(T2{data[i], data[i + 1]});
        float2 b = Converter::convert(T2{other.data[i], other.data[i + 1]});
        a.x *= b.x;
        a.y *= b.y;
        T2 temp = Converter::convert(a);
        data[i] = temp.x;
        data[i + 1] = temp.y;
      }
    } else {
      #pragma unroll
      for (int i = 0; i < width; ++i) {
        float a = Converter::convert(data[i]);
        float b = Converter::convert(other.data[i]);
        data[i] = Converter::convert(a * b);
      }
    }
    return *this;
  }

  __device__ _f16Vec& operator*=(const float scale) {
    if constexpr (width % 2 == 0) {
      #pragma unroll
      for (int i = 0; i < width; i += 2) {
        float2 temp_f = Converter::convert(T2{data[i], data[i + 1]});
        temp_f.x *= scale;
        temp_f.y *= scale;
        T2 temp = Converter::convert(temp_f);
        data[i] = temp.x;
        data[i + 1] = temp.y;
      }
    } else {
      #pragma unroll
      for (int i = 0; i < width; ++i) {
        float temp = Converter::convert(data[i]) * scale;
        data[i] = Converter::convert(temp);
      }
    }
    return *this;
  }

  __device__ float sum_squares() const {
    float result = 0.0f;
    if constexpr (width % 2 == 0) {
      #pragma unroll
      for (int i = 0; i < width; i += 2) {
        float2 z = Converter::convert(T2{data[i], data[i + 1]});
        result += z.x * z.x + z.y * z.y;
      }
    } else {
      #pragma unroll
      for (int i = 0; i < width; ++i) {
        float x = Converter::convert(data[i]);
        result += x * x;
      }
    }
    return result;
  }

  __device__ float sum() const {
    float result = 0.0f;
    if constexpr (width % 2 == 0) {
      #pragma unroll
      for (int i = 0; i < width; i += 2) {
        float2 z = Converter::convert(T2{data[i], data[i + 1]});
        result += z.x + z.y;
      }
    } else {
      #pragma unroll
      for (int i = 0; i < width; ++i) {
        result += Converter::convert(data[i]);
      }
    }
    return result;
  }

  __device__ float sum_squared_diff(float mean) const {
    float result = 0.0f;
    if constexpr (width % 2 == 0) {
      #pragma unroll
      for (int i = 0; i < width; i += 2) {
        float2 z = Converter::convert(T2{data[i], data[i + 1]});
        float dx = z.x - mean;
        float dy = z.y - mean;
        result += dx * dx + dy * dy;
      }
    } else {
      #pragma unroll
      for (int i = 0; i < width; ++i) {
        float x = Converter::convert(data[i]);
        float d = x - mean;
        result += d * d;
      }
    }
    return result;
  }

  __device__ void normalize(float mean, float scale) {
    if constexpr (width % 2 == 0) {
      #pragma unroll
      for (int i = 0; i < width; i += 2) {
        // Convert to float2 for higher precision arithmetic
        float2 z = Converter::convert(T2{data[i], data[i + 1]});
        // Do normalization in float32
        z.x = (z.x - mean) * scale;
        z.y = (z.y - mean) * scale;
        // Convert back to half/bfloat16
        T2 temp = Converter::convert(z);
        data[i] = temp.x;
        data[i + 1] = temp.y;
      }
    } else {
      #pragma unroll
      for (int i = 0; i < width; ++i) {
        float x = Converter::convert(data[i]);
        x = (x - mean) * scale;
        data[i] = Converter::convert(x);
      }
    }
  }
};

/* Function specialization in the case of FP16/BF16 tensors.
   Additional optimizations we can make in this case are
   packed and vectorized operations, which help with the
   memory latency bottleneck. */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,         // [..., hidden_size]
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  // Sanity checks on our vector struct and type-punned pointer arithmetic
  static_assert(std::is_standard_layout_v<_f16Vec<scalar_t, width>> &&
                std::is_trivial_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  __shared__ float s_variance;
  float variance = 0.0f;
  /* These and the argument pointers are all declared `restrict` as they are
     not aliased in practice. Argument pointers should not be dereferenced
     in this kernel as that would be undefined behavior */
  auto* __restrict__ input_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(input);
  auto* __restrict__ residual_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(residual);
  auto* __restrict__ weight_v =
      reinterpret_cast<const _f16Vec<scalar_t, width>*>(weight);

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    _f16Vec<scalar_t, width> temp = input_v[id];
    temp += residual_v[id];
    variance += temp.sum_squares();
    residual_v[id] = temp;
  }
  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    _f16Vec<scalar_t, width> temp = residual_v[id];
    temp *= s_variance;
    temp *= weight_v[idx];
    input_v[id] = temp;
  }
}

/* Generic fused_add_rms_norm_kernel
   The width field is not used here but necessary for other specializations.
 */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,         // [..., hidden_size]
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    scalar_t z = input[blockIdx.x * hidden_size + idx];
    z += residual[blockIdx.x * hidden_size + idx];
    float x = (float)z;
    variance += x * x;
    residual[blockIdx.x * hidden_size + idx] = z;
  }
  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)residual[blockIdx.x * hidden_size + idx];
    input[blockIdx.x * hidden_size + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

template <typename scalar_t>
__global__ void layer_norm_kernel(
    scalar_t* __restrict__ out,           // [..., hidden_size]
    const scalar_t* __restrict__ input,   // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const scalar_t* __restrict__ bias,    // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float)input[blockIdx.x * hidden_size + idx];
    mean += x;
  }
  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  mean = BlockReduce(reduceStore).Reduce(mean, cub::Sum{}, blockDim.x);
  if (threadIdx.x == 0) {
    s_mean = mean / hidden_size;
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float)input[blockIdx.x * hidden_size + idx];
    variance += (x - s_mean) * (x - s_mean);
  }
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)input[blockIdx.x * hidden_size + idx];
    x = (x - s_mean) * s_variance;
    out[blockIdx.x * hidden_size + idx] = 
        ((scalar_t)(x * weight[idx] + bias[idx]));
  }
}

template <typename scalar_t, int width>
__global__ void fused_add_layer_norm_kernel_optimized(
    scalar_t* __restrict__ input,         // [..., hidden_size]
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const scalar_t* __restrict__ bias,    // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  #ifdef __CUDA_ARCH__
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("Running on SM %d\n", __CUDA_ARCH__);
  }
  #endif
  // Move SFINAE check to function level using std::enable_if_t
  static_assert((width > 0) && _typeConvert<scalar_t>::exists,
                "This kernel requires width > 0 and supported data type");

  static_assert(std::is_standard_layout_v<_f16Vec<scalar_t, width>> &&
                std::is_trivial_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  __shared__ float s_mean;
  __shared__ float s_variance;
  float sum = 0.0f;

  auto* __restrict__ input_v = 
      reinterpret_cast<_f16Vec<scalar_t, width>*>(input);
  auto* __restrict__ residual_v = 
      reinterpret_cast<_f16Vec<scalar_t, width>*>(residual);
  auto* __restrict__ weight_v = 
      reinterpret_cast<const _f16Vec<scalar_t, width>*>(weight);
  auto* __restrict__ bias_v = 
      reinterpret_cast<const _f16Vec<scalar_t, width>*>(bias);

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    _f16Vec<scalar_t, width> temp = input_v[id];
    temp += residual_v[id];
    residual_v[id] = temp;
    sum += temp.sum();
  }
  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  sum = BlockReduce(reduceStore).Reduce(sum, cub::Sum{}, blockDim.x);
  if (threadIdx.x == 0) {
    s_mean = sum / hidden_size;
  }
  __syncthreads();

  float var = 0.0f;
  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    _f16Vec<scalar_t, width> temp = residual_v[id];
    var += temp.sum_squared_diff(s_mean);
  }
  var = BlockReduce(reduceStore).Reduce(var, cub::Sum{}, blockDim.x);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(var / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    _f16Vec<scalar_t, width> temp = residual_v[id];
    temp.normalize(s_mean, s_variance);
    temp *= weight_v[idx];
    temp += bias_v[idx];
    input_v[id] = temp;
  }
}

// Add a non-optimized fallback version
template <typename scalar_t, int width>
__global__ void fused_add_layer_norm_kernel_fallback(
    scalar_t* __restrict__ input,         // [..., hidden_size]
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const scalar_t* __restrict__ bias,    // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;

  // First pass: Add residual and compute mean (all in float32)
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)input[blockIdx.x * hidden_size + idx];
    x += (float)residual[blockIdx.x * hidden_size + idx];
    mean += x;
    // Store the float sum back to residual for reuse
    residual[blockIdx.x * hidden_size + idx] = (scalar_t)x;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  mean = BlockReduce(reduceStore).Reduce(mean, cub::Sum{}, blockDim.x);
  if (threadIdx.x == 0) {
    s_mean = mean / hidden_size;
  }
  __syncthreads();

  // Second pass: Compute variance
  float variance = 0.0f;
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)residual[blockIdx.x * hidden_size + idx];
    float diff = x - s_mean;
    variance += diff * diff;
  }
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  // Final pass: Normalize and apply weight/bias
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)residual[blockIdx.x * hidden_size + idx];
    x = (x - s_mean) * s_variance;
    // Apply weight and bias in float32 before final conversion
    x = x * (float)weight[idx] + (float)bias[idx];
    input[blockIdx.x * hidden_size + idx] = (scalar_t)x;
  }
}


template <typename scalar_t>
__global__ void ada_layer_norm_kernel(
    scalar_t* __restrict__ out,           // [..., hidden_size]
    const scalar_t* __restrict__ input,   // [..., hidden_size]
    const scalar_t* __restrict__ gamma,   // [..., hidden_size]
    const scalar_t* __restrict__ beta,    // [..., hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;

  // First pass: compute mean
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)input[blockIdx.x * hidden_size + idx];
    mean += x;
  }
  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  mean = BlockReduce(reduceStore).Reduce(mean, cub::Sum{}, blockDim.x);
  if (threadIdx.x == 0) {
    s_mean = mean / hidden_size;
  }
  __syncthreads();

  // Second pass: compute variance
  float variance = 0.0f;
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)input[blockIdx.x * hidden_size + idx];
    float diff = x - s_mean;
    variance += diff * diff;
  }
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  // Final pass: normalize and apply style-based gamma/beta
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)input[blockIdx.x * hidden_size + idx];
    x = (x - s_mean) * s_variance;
    // Apply style-based gamma (1 + gamma) and beta
    x = x * (1.0f + (float)gamma[blockIdx.x * hidden_size + idx]) + 
        (float)beta[blockIdx.x * hidden_size + idx];
    out[blockIdx.x * hidden_size + idx] = (scalar_t)x;
  }
}


template <typename scalar_t>
__global__ void ada_instance_norm_kernel(
    scalar_t* __restrict__ out,           // [B, C, T]
    const scalar_t* __restrict__ input,   // [B, C, T]
    const scalar_t* __restrict__ gamma,   // [B, C, 1]
    const scalar_t* __restrict__ beta,    // [B, C, 1]
    const float epsilon,
    const int batch_size,
    const int channels,
    const int time_steps) {

    // Each block handles one channel of one batch
    const int batch_idx = blockIdx.x / channels;
    const int channel_idx = blockIdx.x % channels;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;

    // mean across time dimension
    for (int t = threadIdx.x; t < time_steps; t += blockDim.x) {
        const int idx = (batch_idx * channels + channel_idx) * time_steps + t;
        mean += (float)input[idx];
    }

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage reduceStore;
    mean = BlockReduce(reduceStore).Reduce(mean, cub::Sum{}, blockDim.x);
    if (threadIdx.x == 0) {
        s_mean = mean / time_steps;
    }
    __syncthreads();

    // variance
    float variance = 0.0f;
    for (int t = threadIdx.x; t < time_steps; t += blockDim.x) {
        const int idx = (batch_idx * channels + channel_idx) * time_steps + t;
        float diff = (float)input[idx] - s_mean;
        variance += diff * diff;
    }
    variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / time_steps + epsilon);
    }
    __syncthreads();

    // normalization and style-based scaling
    const float g = 1.0f + (float)gamma[batch_idx * channels + channel_idx];
    const float b = (float)beta[batch_idx * channels + channel_idx];

    for (int t = threadIdx.x; t < time_steps; t += blockDim.x) {
        const int idx = (batch_idx * channels + channel_idx) * time_steps + t;
        float x = (float)input[idx];
        x = (x - s_mean) * s_variance;
        out[idx] = (scalar_t)(g * x + b);
    }
}

}  // namespace kizuna

void rms_norm(torch::Tensor& out,     // [..., hidden_size]
              torch::Tensor& input,   // [..., hidden_size]
              torch::Tensor& weight,  // [hidden_size]
              double epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  KIZUNA_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "rms_norm_kernel", [&] {
        kizuna::rms_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(), epsilon, num_tokens, hidden_size);
      });
}

#define LAUNCH_FUSED_ADD_RMS_NORM(width)                                       \
  KIZUNA_DISPATCH_FLOATING_TYPES(                                              \
      input.scalar_type(), "fused_add_rms_norm_kernel", [&] {                  \
        kizuna::fused_add_rms_norm_kernel<scalar_t, width>                     \
            <<<grid, block, 0, stream>>>(input.data_ptr<scalar_t>(),           \
                                         residual.data_ptr<scalar_t>(),        \
                                         weight.data_ptr<scalar_t>(), epsilon, \
                                         num_tokens, hidden_size);             \
      });

void fused_add_rms_norm(torch::Tensor& input,     // [..., hidden_size]
                        torch::Tensor& residual,  // [..., hidden_size]
                        torch::Tensor& weight,    // [hidden_size]
                        double epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops. */
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 block(std::min(hidden_size, max_block_size));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  /*If the tensor types are FP16/BF16, try to use the optimized kernel
    with packed + vectorized ops.
    Max optimization is achieved with a width-8 vector of FP16/BF16s
    since we can load at most 128 bits at once in a global memory op.
    However, this requires each tensor's data to be aligned to 16
    bytes.
   */
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  bool ptrs_are_aligned =
      inp_ptr % 16 == 0 && res_ptr % 16 == 0 && wt_ptr % 16 == 0;
  if (ptrs_are_aligned && hidden_size % 8 == 0) {
    LAUNCH_FUSED_ADD_RMS_NORM(8);
  } else {
    LAUNCH_FUSED_ADD_RMS_NORM(0);
  }
}


void layer_norm(torch::Tensor& out,      // [..., hidden_size]
                torch::Tensor& input,    // [..., hidden_size]
                torch::Tensor& weight,   // [hidden_size]
                torch::Tensor& bias,     // [hidden_size]
                double epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  KIZUNA_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "layer_norm_kernel", [&] {
        kizuna::layer_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
            epsilon, num_tokens, hidden_size);
      });
}

#define LAUNCH_FUSED_ADD_LAYER_NORM(width)                                     \
  KIZUNA_DISPATCH_FLOATING_TYPES(                                             \
      input.scalar_type(), "fused_add_layer_norm_kernel", [&] {               \
        if constexpr ((width > 0) && kizuna::_typeConvert<scalar_t>::exists) {\
          kizuna::fused_add_layer_norm_kernel_optimized<scalar_t, width>      \
              <<<grid, block, 0, stream>>>(input.data_ptr<scalar_t>(),        \
                                          residual.data_ptr<scalar_t>(),       \
                                          weight.data_ptr<scalar_t>(),         \
                                          bias.data_ptr<scalar_t>(),           \
                                          epsilon, num_tokens, hidden_size);    \
        } else {                                                               \
          kizuna::fused_add_layer_norm_kernel_fallback<scalar_t, width>       \
              <<<grid, block, 0, stream>>>(input.data_ptr<scalar_t>(),        \
                                          residual.data_ptr<scalar_t>(),       \
                                          weight.data_ptr<scalar_t>(),         \
                                          bias.data_ptr<scalar_t>(),           \
                                          epsilon, num_tokens, hidden_size);    \
        }                                                                      \
      });

void fused_add_layer_norm(torch::Tensor& input,     // [..., hidden_size]
                         torch::Tensor& residual,   // [..., hidden_size]
                         torch::Tensor& weight,     // [hidden_size]
                         torch::Tensor& bias,       // [hidden_size]
                         double epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 block(std::min(hidden_size, max_block_size));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  auto bias_ptr = reinterpret_cast<std::uintptr_t>(bias.data_ptr());
  bool ptrs_are_aligned =
      inp_ptr % 16 == 0 && res_ptr % 16 == 0 && 
      wt_ptr % 16 == 0 && bias_ptr % 16 == 0;
  if (ptrs_are_aligned && hidden_size % 8 == 0) {
    LAUNCH_FUSED_ADD_LAYER_NORM(8);
  } else {
    LAUNCH_FUSED_ADD_LAYER_NORM(0);
  }
}

void ada_layer_norm(
    torch::Tensor& out,      // [..., hidden_size]
    torch::Tensor& input,    // [..., hidden_size]
    torch::Tensor& gamma,    // [..., hidden_size]
    torch::Tensor& beta,     // [..., hidden_size]
    double epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  KIZUNA_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "ada_layer_norm_kernel", [&] {
        kizuna::ada_layer_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(), beta.data_ptr<scalar_t>(),
            epsilon, num_tokens, hidden_size);
      });
}

void ada_instance_norm(
    torch::Tensor& out,      // [B, C, T]
    torch::Tensor& input,    // [B, C, T]
    torch::Tensor& gamma,    // [B, C, 1]
    torch::Tensor& beta,     // [B, C, 1]
    double epsilon) {
    
    int batch_size = input.size(0);
    int channels = input.size(1);
    int time_steps = input.size(2);

    dim3 grid(batch_size * channels);  // One block per (batch, channel) pair
    dim3 block(std::min(time_steps, 1024));
    
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    KIZUNA_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "ada_instance_norm_kernel", [&] {
            kizuna::ada_instance_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                gamma.data_ptr<scalar_t>(),
                beta.data_ptr<scalar_t>(),
                epsilon,
                batch_size,
                channels,
                time_steps
            );
        });
}
