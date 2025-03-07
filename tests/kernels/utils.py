from typing import Optional, Tuple, Union

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


class AdaLayerNorm(torch.nn.Module):
    """Reference implementation of adaptive layer norm."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.variance_epsilon = eps

    def forward_native(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,  # [..., hidden_size]
        beta: torch.Tensor,  # [..., hidden_size]
    ) -> torch.Tensor:
        """PyTorch reference implementation."""
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        gamma = gamma.to(torch.float32)
        beta = beta.to(torch.float32)

        mean = x.mean(dim=-1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.variance_epsilon)
        return (x * (1 + gamma) + beta).to(orig_dtype)

    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Custom CUDA kernel implementation."""
        from kizuna import _custom_ops as ops

        out = torch.empty_like(x)
        ops.ada_layer_norm(
            out,
            x,
            gamma,
            beta,
            self.variance_epsilon,
        )
        return out


class AdaInstanceNorm(torch.nn.Module):
    """Reference implementation of adaptive instance norm."""

    def __init__(
        self,
        style_dim: int,
        num_features: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(style_dim, num_features * 2)
        self.variance_epsilon = eps

    def forward_native(
        self,
        x: torch.Tensor,  # [B, C, T]
        s: torch.Tensor,  # [B, style_dim]
    ) -> torch.Tensor:
        """PyTorch reference implementation."""
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        s = s.to(torch.float32)

        orig_weight = self.fc.weight.data
        orig_bias = self.fc.bias.data
        self.fc.weight.data = orig_weight.to(torch.float32)
        self.fc.bias.data = orig_bias.to(torch.float32)

        try:
            h = self.fc(s)
            h = h.view(h.size(0), h.size(1), 1)
            gamma, beta = torch.chunk(h, chunks=2, dim=1)

            mean = x.mean(dim=-1, keepdim=True)
            variance = (x - mean).pow(2).mean(dim=-1, keepdim=True)
            x = (x - mean) * torch.rsqrt(variance + self.variance_epsilon)
            return (x * (1 + gamma) + beta).to(orig_dtype)
        finally:
            self.fc.weight.data = orig_weight
            self.fc.bias.data = orig_bias

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        """Custom CUDA kernel implementation."""
        from kizuna import _custom_ops as ops

        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        out = torch.empty_like(x)
        ops.ada_instance_norm(
            out,
            x,
            gamma,
            beta,
            self.variance_epsilon,
        )
        return out
