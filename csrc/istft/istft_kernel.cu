#include <c10/cuda/CUDAGuard.h>

#include "istft_kernel.h"
#include "static_switch.h"
#include "../dispatch_utils.h"

namespace kizuna {

template<typename Ktraits>
__global__ void istft_kernel(ISTFTParams params) {
    using input_t = typename Ktraits::input_t;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNFFT = Ktraits::kNFFT;
    constexpr int kBinsPerThread = Ktraits::kBinsPerThread;

    extern __shared__ char smem_[];
    float2* smem = reinterpret_cast<float2*>(smem_);
    
    const int batch_idx = blockIdx.x;
    const int frame_start = blockIdx.y * Ktraits::kMaxFramesPerBlock;
    const int tid = threadIdx.x;

    if (tid < params.win_length) {
        float window_val = reinterpret_cast<float*>(params.window_ptr)[tid];
        smem[tid].x = window_val;
        smem[tid].y = 0.0f;
    }
    __syncthreads();

    for (int frame_offset = 0; frame_offset < Ktraits::kMaxFramesPerBlock; frame_offset++) {
        const int frame_idx = frame_start + frame_offset;
        if (frame_idx >= params.n_frames) break;

        float2 complex_spec[kBinsPerThread];
        #pragma unroll
        for (int i = 0; i < kBinsPerThread; i++) {
            const int bin = tid * kBinsPerThread + i;
            if (bin <= kNFFT/2) {
                const int64_t offset = batch_idx * params.mag_stride + frame_idx * (kNFFT/2 + 1) + bin;
                if (offset < params.mag_stride * params.batch_size) {
                    float mag = reinterpret_cast<input_t*>(params.mag_ptr)[offset];
                    float phase = reinterpret_cast<input_t*>(params.phase_ptr)[offset];
                    complex_spec[i].x = mag * cosf(phase);
                    complex_spec[i].y = mag * sinf(phase);
                } else {
                    complex_spec[i].x = complex_spec[i].y = 0.0f;
                }
            } else {
                complex_spec[i].x = complex_spec[i].y = 0.0f;
            }
        }
        __syncthreads();

        for (int stage = 0; stage < log2f(kNFFT); stage++) {
            const int butterfly_size = 1 << stage;
            const int butterfly_stride = butterfly_size * 2;

            #pragma unroll
            for (int i = 0; i < kBinsPerThread; i++) {
                const int bin = tid * kBinsPerThread + i;
                if (bin < kNFFT) {
                    const int pair_idx = (bin & (butterfly_size - 1)) + ((bin >> stage) << (stage + 1));
                    if (pair_idx < kNFFT) {
                        const float angle = -2.0f * M_PI * (bin & (butterfly_size - 1)) / butterfly_stride;
                        const float2 twiddle = make_float2(cosf(angle), sinf(angle));

                        float2 pair;
                        pair.x = __shfl_xor_sync(FULL_MASK, complex_spec[i].x, butterfly_size);
                        pair.y = __shfl_xor_sync(FULL_MASK, complex_spec[i].y, butterfly_size);

                        const float2 temp = complex_spec[i];
                        complex_spec[i].x = temp.x + (twiddle.x * pair.x - twiddle.y * pair.y);
                        complex_spec[i].y = temp.y + (twiddle.x * pair.y + twiddle.y * pair.x);
                    }
                }
            }
            __syncthreads();
        }

        const int output_offset = frame_idx * params.hop_length;
        #pragma unroll
        for (int i = 0; i < kBinsPerThread; i++) {
            const int bin = tid * kBinsPerThread + i;
            if (bin < params.win_length) {
                const float window_val = smem[bin].x;
                const float real_val = complex_spec[i].x * window_val;
                const int64_t out_idx = batch_idx * params.out_stride + output_offset + bin;
                if (output_offset + bin < params.out_length && 
                    out_idx < params.out_stride * params.batch_size) {
                    atomicAdd(&reinterpret_cast<float*>(params.out_ptr)[out_idx], real_val);
                }
            }
        }
        __syncthreads();
    }
}

template<typename input_t>
void launch_istft_kernel(ISTFTParams& params, cudaStream_t stream) {
    constexpr int kNThreads = 256;

    int n_fft_pow2 = 32;
    while (n_fft_pow2 < params.n_fft) {
        n_fft_pow2 *= 2;
    }

    BOOL_SWITCH(n_fft_pow2 <= 32, N32, [&] {
        BOOL_SWITCH(n_fft_pow2 <= 64 && n_fft_pow2 > 32, N64, [&] {
            BOOL_SWITCH(n_fft_pow2 <= 128 && n_fft_pow2 > 64, N128, [&] {
                BOOL_SWITCH(n_fft_pow2 <= 256 && n_fft_pow2 > 128, N256, [&] {
                    BOOL_SWITCH(n_fft_pow2 <= 512 && n_fft_pow2 > 256, N512, [&] {
                        BOOL_SWITCH(n_fft_pow2 <= 1024 && n_fft_pow2 > 512, N1024, [&] {
                            BOOL_SWITCH(n_fft_pow2 <= 2048 && n_fft_pow2 > 1024, N2048, [&] {
                                BOOL_SWITCH(n_fft_pow2 <= 4096 && n_fft_pow2 > 2048, N4096, [&] {
                                    BOOL_SWITCH(n_fft_pow2 <= 8192 && n_fft_pow2 > 4096, N8192, [&] {
                                        using Ktraits = ISTFTKernelTraits<kNThreads,
                                            N32 ? 32 : N64 ? 64 : N128 ? 128 : N256 ? 256 : 
                                            N512 ? 512 : N1024 ? 1024 : N2048 ? 2048 : 
                                            N4096 ? 4096 : 8192,
                                            input_t>;

                                        const int n_frame_blocks = (params.n_frames + Ktraits::kMaxFramesPerBlock - 1) / 
                                                                 Ktraits::kMaxFramesPerBlock;
                                        dim3 grid(params.batch_size, n_frame_blocks);
                                        dim3 block(kNThreads);

                                        istft_kernel<Ktraits><<<grid, block, Ktraits::kSmemSize, stream>>>(params);
                                    });
                                });
                            });
                        });
                    });
                });
            });
        });
    });

    if (n_fft_pow2 > 8192) {
        throw std::runtime_error("FFT size too large (max 8192)");
    }
    
    C10_CUDA_CHECK(cudaGetLastError());
}

} // namespace kizuna

void istft(torch::Tensor& output,
           torch::Tensor& magnitude,
           torch::Tensor& phase,
           torch::Tensor& window,
           int64_t hop_length,
           bool center,
           bool normalized) {

    TORCH_CHECK(magnitude.is_cuda(), "Magnitude must be a CUDA tensor");
    TORCH_CHECK(phase.is_cuda(), "Phase must be a CUDA tensor");
    TORCH_CHECK(window.is_cuda(), "Window must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "Output must be a CUDA tensor");

    TORCH_CHECK(magnitude.dim() == 3, "Magnitude must be 3D (batch, freq, time)");
    TORCH_CHECK(phase.dim() == 3, "Phase must be 3D (batch, freq, time)");
    TORCH_CHECK(window.dim() == 1, "Window must be 1D");

    const int batch_size = magnitude.size(0);
    const int n_fft = (magnitude.size(1) - 1) * 2;
    const int n_frames = magnitude.size(2);
    const int win_length = window.size(0);

    kizuna::ISTFTParams params;
    params.batch_size = batch_size;
    params.n_fft = n_fft;
    params.hop_length = hop_length;
    params.win_length = win_length;
    params.n_frames = n_frames;

    params.mag_stride = magnitude.stride(0);
    params.phase_stride = phase.stride(0);
    params.out_stride = output.stride(0);

    params.mag_ptr = magnitude.data_ptr();
    params.phase_ptr = phase.data_ptr();
    params.window_ptr = window.data_ptr();
    params.out_ptr = output.data_ptr();

    params.out_length = output.size(1);

    auto stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        magnitude.scalar_type(), "istft", ([&] {
            kizuna::launch_istft_kernel<scalar_t>(params, stream);
        }));
}
