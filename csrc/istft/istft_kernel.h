#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define FULL_MASK 0xffffffff

// 8 uint4's = 32 bytes
struct uint8 {
  uint4 u;
  uint4 v;
};

template <int BYTES> struct BytesToType {};

template <> struct BytesToType<32> {
  using Type = uint8;
};
template <> struct BytesToType<16> {
  using Type = uint4;
};
template <> struct BytesToType<8> {
  using Type = uint64_t;
};
template <> struct BytesToType<4> {
  using Type = uint32_t;
};
template <> struct BytesToType<2> {
  using Type = uint16_t;
};
template <> struct BytesToType<1> {
  using Type = uint8_t;
};

namespace kizuna {

struct ISTFTParams {
  // Dimensions
  int batch_size; // Batch size
  int n_fft;      // FFT size
  int hop_length; // Hop length between frames
  int win_length; // Window length
  int n_frames;   // Number of time frames

  // Strides
  int64_t mag_stride;   // Magnitude spectrogram stride
  int64_t phase_stride; // Phase spectrogram stride
  int64_t out_stride;   // Output audio stride

  // Pointers
  void *mag_ptr;    // Magnitude spectrogram (B, F, T)
  void *phase_ptr;  // Phase spectrogram (B, F, T)
  void *window_ptr; // Window function (win_length)
  void *out_ptr;    // Output audio

  // Output length
  int out_length; // Length of output audio
};

template <int kNThreads_, int kNFFT_, typename input_t_>
struct ISTFTKernelTraits {
  using input_t = input_t_;
  static constexpr int kNThreads = kNThreads_;
  static constexpr int kNFFT = kNFFT_;

  static constexpr int kBinsPerThread =
      (kNFFT / 2 + 1 + kNThreads - 1) / kNThreads;

  static constexpr int kVecSize = 4;
  using vec_t = typename BytesToType<sizeof(input_t) * kVecSize>::Type;

  static constexpr int kSmemSize = kNFFT * sizeof(float2);

  static constexpr int kMaxFramesPerBlock = 8;
};

template <typename Ktraits> __global__ void istft_kernel(ISTFTParams params);

template <typename input_t>
void launch_istft_kernel(ISTFTParams &params, cudaStream_t stream);

} // namespace kizuna
