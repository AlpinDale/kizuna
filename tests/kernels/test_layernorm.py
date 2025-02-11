import pytest
import torch

from kizuna.modeling.istftnet import AdaIN1d

from .utils import AdaLayerNorm, LayerNorm, RMSNorm

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
    ref_out = layer.forward_native(
        x.clone(), residual.clone() if residual is not None else None
    )
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
            print(
                f"residual: mean={residual.mean().item():.6f}, std={residual.std().item():.6f}"
            )
        print(
            f"weight: mean={layer.weight.mean().item():.6f}, std={layer.weight.std().item():.6f}"
        )
        print(
            f"bias: mean={layer.bias.mean().item():.6f}, std={layer.bias.std().item():.6f}"
        )

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
        print(
            f"ref_out: mean={ref_out[0].mean().item():.6f}, std={ref_out[0].std().item():.6f}"
        )
        print(
            f"cuda_out: mean={out[0].mean().item():.6f}, std={out[0].std().item():.6f}"
        )

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


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_ada_layer_norm(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    """Test adaptive layer normalization."""
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)

    layer = AdaLayerNorm(hidden_size).to(dtype=dtype)
    scale = 1 / (2 * hidden_size)

    x = torch.randn(num_tokens, hidden_size, dtype=dtype) * scale
    gamma = torch.randn(num_tokens, hidden_size, dtype=dtype) * 0.1
    beta = torch.randn(num_tokens, hidden_size, dtype=dtype) * 0.1

    # Reference implementation should be executed first
    ref_out = layer.forward_native(x.clone(), gamma.clone(), beta.clone())
    out = layer(x, gamma, beta)

    if dtype == torch.float16 and hidden_size == 768 and num_tokens == 7:
        print("\nInput stats:")
        print(f"x: mean={x.mean().item():.6f}, std={x.std().item():.6f}")
        print(f"gamma: mean={gamma.mean().item():.6f}, std={gamma.std().item():.6f}")
        print(f"beta: mean={beta.mean().item():.6f}, std={beta.std().item():.6f}")
        print("\nOutput stats:")
        print(
            f"ref_out: mean={ref_out.mean().item():.6f}, std={ref_out.std().item():.6f}"
        )
        print(f"cuda_out: mean={out.mean().item():.6f}, std={out.std().item():.6f}")

        diff = (out - ref_out).abs()
        max_diff_idx = diff.argmax()
        print(f"\nLargest difference at index {max_diff_idx.item()}:")
        print(f"ref_value: {ref_out.flatten()[max_diff_idx].item():.6f}")
        print(f"cuda_value: {out.flatten()[max_diff_idx].item():.6f}")
        print(f"abs_diff: {diff.max().item():.6f}")

    # Use higher tolerance for bfloat16
    if dtype == torch.bfloat16:
        torch.testing.assert_close(out, ref_out, atol=5e-2, rtol=5e-2)
    else:
        torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("channels", [32, 64, 128])
@pytest.mark.parametrize("time_steps", [16, 128, 512])
@pytest.mark.parametrize("style_dim", [64, 128])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_ada_instance_norm(
    batch_size: int,
    channels: int,
    time_steps: int,
    style_dim: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    """Test adaptive instance normalization."""
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)

    layer = AdaIN1d(style_dim, channels)
    # Initialize weights and convert to target dtype
    layer.fc.weight.data.normal_(mean=0.0, std=0.02)
    layer.fc.bias.data.zero_()
    layer = layer.to(dtype=dtype)

    x = torch.randn(batch_size, channels, time_steps, dtype=dtype)
    s = torch.randn(batch_size, style_dim, dtype=dtype)

    # Original PyTorch implementation
    x_ref = x.clone()
    s_ref = s.clone()
    ref_out = layer.forward(x_ref, s_ref)  # Original implementation

    # CUDA implementation
    x_cuda = x.clone()
    s_cuda = s.clone()
    out = layer.forward_cuda(x_cuda, s_cuda)  # Our CUDA implementation

    if dtype == torch.float16 and channels == 32 and time_steps == 16:
        print("\nInput stats:")
        print(f"x: mean={x.mean().item():.6f}, std={x.std().item():.6f}")
        print(f"s: mean={s.mean().item():.6f}, std={s.std().item():.6f}")
        print("\nOutput stats:")
        print(
            f"ref_out: mean={ref_out.mean().item():.6f}, std={ref_out.std().item():.6f}"
        )
        print(f"cuda_out: mean={out.mean().item():.6f}, std={out.std().item():.6f}")

        diff = (out - ref_out).abs()
        max_diff_idx = diff.argmax()
        print(f"\nLargest difference at index {max_diff_idx.item()}:")
        print(f"ref_value: {ref_out.flatten()[max_diff_idx].item():.6f}")
        print(f"cuda_value: {out.flatten()[max_diff_idx].item():.6f}")
        print(f"abs_diff: {diff.max().item():.6f}")

    # Use higher tolerance for bfloat16
    if dtype == torch.bfloat16:
        torch.testing.assert_close(out, ref_out, atol=5e-2, rtol=5e-2)
    else:
        torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)
