# https://github.com/yl4579/StyleTTS2/blob/main/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrizations
from transformers import AlbertModel

from kizuna import _custom_ops as ops
from kizuna.modeling.custom_op import CustomOp

from .istftnet import AdainResBlk1d


class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        return self.linear_layer(x)


class LayerNorm(CustomOp):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward_native(self, x, residual=None):
        """PyTorch reference implementation."""
        if residual is not None:
            x = x + residual
            residual = x  # Save for return

        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        x = x.transpose(1, -1)

        if residual is not None:
            return x, residual

        return x

    def forward_cuda(self, x, residual=None):
        """Custom CUDA kernel implementation."""
        if residual is not None:
            x = x + residual
            residual = x  # Save for return

        x = x.transpose(1, -1)  # move channels to last dimension
        out = torch.empty_like(x)
        ops.layer_norm(
            out,
            x,
            self.gamma.data,
            self.beta.data,
            self.eps,
        )
        out = out.transpose(1, -1)  # move them back

        if residual is not None:
            return out, residual

        return out


class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(
                nn.Sequential(
                    parametrizations.weight_norm(
                        nn.Conv1d(
                            channels, channels, kernel_size=kernel_size, padding=padding
                        )
                    ),
                    LayerNorm(channels),
                    actv,
                    nn.Dropout(0.2),
                )
            )
        self.lstm = nn.LSTM(
            channels,
            channels // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

    def forward(self, x, input_lengths, m):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.to(input_lengths.device).unsqueeze(1)
        x.masked_fill_(m, 0.0)
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
        x = x.transpose(1, 2)  # [B, T, chn]
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False
        )
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x.transpose(-1, -2)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])
        x_pad[:, :, : x.shape[-1]] = x
        x = x_pad.to(x.device)
        x.masked_fill_(m, 0.0)
        return x


class AdaLayerNorm(CustomOp):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward_native(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)

    def forward_cuda(self, x, s):
        # Original dimension order: [B, C, T]
        x = x.transpose(-1, -2)  # -> [B, T, C]
        x = x.transpose(1, -1)  # -> [C, B, T]

        h = self.fc(s)  # [B, channels*2]
        h = h.view(h.size(0), h.size(1), 1)  # [B, channels*2, 1]
        gamma, beta = torch.chunk(h, chunks=2, dim=1)  # Each [B, channels, 1]
        gamma = gamma.transpose(1, -1)  # [1, B, channels]
        beta = beta.transpose(1, -1)  # [1, B, channels]

        gamma = gamma.expand_as(x)  # [C, B, T]
        beta = beta.expand_as(x)  # [C, B, T]

        out = torch.empty_like(x)
        ops.ada_layer_norm(
            out,
            x,
            gamma,
            beta,
            self.eps,
        )

        return out.transpose(1, -1).transpose(-1, -2)  # [B, C, T]


class ProsodyPredictor(nn.Module):
    def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
        super().__init__()
        self.text_encoder = DurationEncoder(
            sty_dim=style_dim, d_model=d_hid, nlayers=nlayers, dropout=dropout
        )
        self.lstm = nn.LSTM(
            d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True
        )
        self.duration_proj = LinearNorm(d_hid, max_dur)
        self.shared = nn.LSTM(
            d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True
        )
        self.F0 = nn.ModuleList()
        self.F0.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.F0.append(
            AdainResBlk1d(
                d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout
            )
        )
        self.F0.append(
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout)
        )
        self.N = nn.ModuleList()
        self.N.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.N.append(
            AdainResBlk1d(
                d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout
            )
        )
        self.N.append(
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout)
        )
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

    def forward(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)
        input_lengths = text_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            d, input_lengths, batch_first=True, enforce_sorted=False
        )
        m = m.to(text_lengths.device).unsqueeze(1)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]])
        x_pad[:, : x.shape[1], :] = x
        x = x_pad.to(x.device)
        duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=False))
        en = d.transpose(-1, -2) @ alignment
        return duration.squeeze(-1), en

    def F0Ntrain(self, x, s):
        x, _ = self.shared(x.transpose(-1, -2))
        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)
        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)
        return F0.squeeze(1), N.squeeze(1)


class DurationEncoder(nn.Module):
    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(
                nn.LSTM(
                    d_model + sty_dim,
                    d_model // 2,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout,
                )
            )
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))
        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward(self, x, style, text_lengths, m):
        masks = m.to(text_lengths.device)
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)
        input_lengths = text_lengths.cpu().numpy()
        x = x.transpose(-1, -2)
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, -1, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                x = x.transpose(-1, -2)
                x = nn.utils.rnn.pack_padded_sequence(
                    x, input_lengths, batch_first=True, enforce_sorted=False
                )
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=False)
                x = x.transpose(-1, -2)
                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])
                x_pad[:, :, : x.shape[-1]] = x
                x = x_pad.to(x.device)
        return x.transpose(-1, -2)


# https://github.com/yl4579/StyleTTS2/blob/main/Utils/PLBERT/util.py
class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        return outputs.last_hidden_state
