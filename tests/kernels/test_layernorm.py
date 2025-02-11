from typing import Optional, Tuple, Union

import pytest
import torch


class RMSNorm(torch.nn.Module):
    """Root mean square normalization."""
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """PyTorch reference implementation."""
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        if residual is None:
            return x
        else:
            return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Custom CUDA kernel implementation."""
        from kizuna import _custom_ops as ops
        if residual is not None:
            ops.fused_add_rms_norm(
                x,
                residual,
                self.weight.data,
                self.variance_epsilon,
            )
            return x, residual
        out = torch.empty_like(x)
        ops.rms_norm(
            out,
            x,
            self.weight.data,
            self.variance_epsilon,
        )
        return out


class LayerNorm(torch.nn.Module):
    """Layer normalization."""
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """PyTorch reference implementation."""
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        mean = x.mean(dim=-1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight + self.bias
        if residual is None:
            return x
        else:
            return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Custom CUDA kernel implementation."""
        from kizuna import _custom_ops as ops
        if residual is not None:
            ops.fused_add_layer_norm(
                x,
                residual,
                self.weight.data,
                self.bias.data,
                self.variance_epsilon,
            )
            return x, residual
        out = torch.empty_like(x)
        ops.layer_norm(
            out,
            x,
            self.weight.data,
            self.bias.data,
            self.variance_epsilon,
        )
        return out


# Test configurations
DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 4096]  # Arbitrary values for testing
HIDDEN_SIZES = [768, 769, 770, 771, 5120, 5124, 5125, 5126, 8192, 8199]
ADD_RESIDUAL = [False, True]
SEEDS = [0]
CUDA_DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_rms_norm(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)

    layer = RMSNorm(hidden_size).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    x *= scale
    residual = torch.randn_like(x) * scale if add_residual else None

    # Reference implementation should be executed first
    # because the custom kernel is in-place
    ref_out = layer.forward_native(x.clone(), residual.clone() if residual is not None else None)
    out = layer(x, residual)

    # LayerNorm operators typically have larger numerical errors
    # Use larger tolerance
    if add_residual:
        torch.testing.assert_close(out[0], ref_out[0], atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out[1], ref_out[1], atol=1e-2, rtol=1e-2)
    else:
        torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_layer_norm(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    
    layer = LayerNorm(hidden_size).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    layer.bias.data.normal_(mean=0.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    x *= scale
    residual = torch.randn_like(x) * scale if add_residual else None

    if dtype == torch.float16 and hidden_size == 768 and num_tokens == 7:
        print("\nInput stats:")
        print(f"x: mean={x.mean().item():.6f}, std={x.std().item():.6f}")
        if residual is not None:
            print(f"residual: mean={residual.mean().item():.6f}, std={residual.std().item():.6f}")
        print(f"weight: mean={layer.weight.mean().item():.6f}, std={layer.weight.std().item():.6f}")
        print(f"bias: mean={layer.bias.mean().item():.6f}, std={layer.bias.std().item():.6f}")

    x_ref = x.clone()
    residual_ref = residual.clone() if residual is not None else None
    x_cuda = x.clone()
    residual_cuda = residual.clone() if residual is not None else None

    # Reference implementation
    ref_out = layer.forward_native(x_ref, residual_ref)

    # CUDA implementation
    out = layer(x_cuda, residual_cuda)

    if dtype == torch.float16 and hidden_size == 768 and num_tokens == 7:
        print("\nOutput stats:")
        print(f"ref_out: mean={ref_out[0].mean().item():.6f}, std={ref_out[0].std().item():.6f}")
        print(f"cuda_out: mean={out[0].mean().item():.6f}, std={out[0].std().item():.6f}")

        diff = (out[0] - ref_out[0]).abs()
        max_diff_idx = diff.argmax()
        print(f"\nLargest difference at index {max_diff_idx.item()}:")
        print(f"ref_value: {ref_out[0].flatten()[max_diff_idx].item():.6f}")
        print(f"cuda_value: {out[0].flatten()[max_diff_idx].item():.6f}")
        print(f"abs_diff: {diff.max().item():.6f}")

    # LayerNorm operators typically have larger numerical errors
    if add_residual:
        # Use higher tolerance for bfloat16
        if dtype == torch.bfloat16:
            torch.testing.assert_close(out[0], ref_out[0], atol=5e-2, rtol=5e-2)
        else:
            torch.testing.assert_close(out[0], ref_out[0], atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out[1], ref_out[1], atol=1e-2, rtol=1e-2)
    else:
        torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)
