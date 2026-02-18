import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class StridingSubsampling(nn.Module):
    """
    Strided convolution subsampling layer.
    Subsampling factor = 4 (stride=2 applied twice).
    Linear projection is done externally.
    """

    def __init__(
        self,
        out_channels: int = 512,
    ):
        super().__init__()
        in_channels = 1
        self.s = 2
        self.ks = 3
        self.p = (self.ks - 1) // 2

        self.conv = nn.Sequential(
                    nn.Conv2d(
                        in_channels, out_channels,
                        kernel_size=self.ks,
                        stride=self.s,
                        padding=self.p,
                        bias=True,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        out_channels, out_channels,
                        kernel_size=self.ks,
                        stride=self.s,
                        padding=self.p,
                        bias=True,
                    ),
                    nn.ReLU(inplace=True),
                )

    def calc_output_length(self, lengths: Tensor) -> Tensor:
        # Length after two stride-2 convolutions
        l = (lengths + 3) // 4
        l = torch.clamp_min(l, 1)
        return l

    def forward(self, x: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor]:
        x = x.unsqueeze(1)          # [B, T, F] -> [B, 1,   T, F]
        x = self.conv(x)            #           -> [B, out_channels(512), ~T/4, F/4]
        b, c, t, f = x.size()
        x = x.transpose(1, 2).contiguous().view(b, t, c * f) # [B, 512, T', 20] -> [B, T', 512, 20] -> [B, T', 512*20]
        return x, self.calc_output_length(lengths)


class RelPositionalEmbedding(nn.Module):
    def __init__(
        self, dim: int=512,
        base: int = 10000,
        init_len=256
    ):
        super().__init__()
        #print(f"Base:{base}")
        self.dim = dim
        self.base = base
        pe = self.create_pe(init_len, torch.device("cpu"))
        self.register_buffer("pe", pe, persistent=False)

    def create_pe(self, length: int, device: torch.device) -> Tensor:
        # Relative positions: from -(L-1) to +(L-1), total 2L-1
        positions = torch.arange(-length + 1, length, device=device).float()  # (2L-1,)
        pe = torch.zeros(positions.size(0), self.dim, device=device)  #[2L-1, dim]

        i = torch.arange(0, self.dim, 2, device=device).float()
        step = -math.log(self.base) / self.dim
        exponent = i * step
        div_term = torch.exp(exponent)

        angles = positions.unsqueeze(1) * div_term.unsqueeze(0)  # (2L-1, dim/2)

        # Sinusoidal encoding: sin for even indices, cos for odd
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)
        return pe.unsqueeze(0)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        input_len = x.size(1)

        if self.pe.size(1) < 2*input_len - 1:
            self.pe = self.create_pe(input_len, self.pe.device)

        full_len = self.pe.size(1)          # = 2L-1
        center = full_len // 2              # center index (0-offset)
        start = center - (input_len - 1)
        end = center + input_len

        return x, self.pe[:, start:end]


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, n_head: int):
        super().__init__()

        assert input_dim % n_head == 0

        self.d_k = input_dim // n_head
        self.h = n_head

        self.linear_q = nn.Linear(input_dim, input_dim)
        self.linear_k = nn.Linear(input_dim, input_dim)
        self.linear_v = nn.Linear(input_dim, input_dim)
        self.linear_out = nn.Linear(input_dim, input_dim)

    def forward_qkv(self, query: Tensor, key: Tensor, value: Tensor):
        b = query.size(0)

        q = self.linear_q(query).view(b, -1, self.h, self.d_k)
        k = self.linear_k(key).view(b, -1, self.h, self.d_k)
        v = self.linear_v(value).view(b, -1, self.h, self.d_k)

        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def forward_attention(self, value: Tensor, scores: Tensor, mask: Optional[Tensor]):
        b = value.size(0)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, -10000.0)
            attn = torch.softmax(scores, dim=-1)
        else:
            attn = torch.softmax(scores, dim=-1)

        x = torch.matmul(attn, value)

        x = x.transpose(1, 2).reshape(b, -1, self.h * self.d_k)
        return self.linear_out(x)


class RelPositionMultiHeadAttention(MultiHeadAttention):
    def __init__(
        self,
        input_dim: int,
        n_head: int
    ):
        super().__init__(input_dim, n_head)

        self.linear_pos = nn.Linear(input_dim, input_dim, bias=False)

        # Positional bias parameters u and v
        self.pos_bias_u = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.FloatTensor(self.h, self.d_k))

        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x: Tensor) -> Tensor:
        b, h, qlen, pos_len = x.size()
        x = x.view(b, h, pos_len, qlen)
        x = x[:, :, 1:, :]
        x = x.view(b, h, qlen, pos_len - 1)
        return x

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        pos_emb: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        q, k, v = self.forward_qkv(q, k, v)

        #q = q.transpose(1, 2)

        p = self.linear_pos(pos_emb)
        p = p.view(pos_emb.shape[0], -1, self.h, self.d_k).transpose(1, 2)

        #q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        #q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)
        q_with_bias_u  = q + self.pos_bias_u.unsqueeze(0).unsqueeze(2)
        q_with_bias_v  = q + self.pos_bias_v.unsqueeze(0).unsqueeze(2)


        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)

        return self.forward_attention(v, scores, mask)


class MultiHeadedSelfAttentionModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        max_len: int = 1024,
        dropout_p: float = 0.1
    ):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.pos_enc = RelPositionalEmbedding(
            dim=input_dim,
            base=10000,
            init_len=max_len
        )

        self.layer_norm = nn.LayerNorm(input_dim)
        self.attention = RelPositionMultiHeadAttention(input_dim, num_heads)
        self.dropout = nn.Dropout(p=dropout_p)


    def forward(self, inputs: Tensor, att_mask: Optional[Tensor] = None):
        batch_size = inputs.size(0)
        inputs, pos_emb = self.pos_enc(inputs)
        if pos_emb.size(0) == 1 and batch_size > 1:
            pos_emb = pos_emb.expand(batch_size, -1, -1)

        inputs = self.layer_norm(inputs)
        outputs = self.attention(inputs, inputs, inputs, pos_emb=pos_emb, mask=att_mask)
        return self.dropout(outputs)


class ConvolutionModule(nn.Module):
    """
    Conformer convolution module.
    Architecture: LayerNorm -> pointwise -> GLU -> depthwise -> BatchNorm -> Swish -> pointwise -> Dropout
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd"

        self.ln = nn.LayerNorm(in_channels)

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=expansion_factor * in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

        self.glu = nn.GLU(dim=1)

        self.depthwise_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=in_channels,
        )

        self.batch_norm = nn.BatchNorm1d(in_channels)
        self.swish = nn.SiLU()

        self.pointwise_conv2 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        self.dropout = nn.Dropout(dropout_p)


    def forward(self, x: Tensor, pad_mask: Optional[Tensor] = None) -> Tensor:

        x = self.ln(x)                # [B, T, C]

        x = x.transpose(1, 2)         # [B, T, C] -> [B, C, T]
        x = self.pointwise_conv1(x)
        x = self.glu(x)

        # Mask padding positions
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)

        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        return x.transpose(1, 2) # [B, C, T] -> [B, T, C]

class FeedForwardModule(nn.Module):
    def __init__(self,
            encoder_dim: int = 512,
            expansion_factor: int = 4,
            dropout_p: float = 0.1,
    ):
        super().__init__()
        d_ff = encoder_dim * expansion_factor

        self.layer_norm = nn.LayerNorm(encoder_dim)
        self.linear1 = nn.Linear(encoder_dim, d_ff, bias=True)
        self.activation = nn.SiLU()  # Swish
        self.dropout = nn.Dropout(dropout_p)
        self.linear2 = nn.Linear(d_ff, encoder_dim, bias=True)



    def forward(self, x: Tensor) -> Tensor:

        x = self.layer_norm(x)   #x: [B, T, encoder_dim]
        x = self.linear1(x)      # expand
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)      # project back
        x = self.dropout(x)
        return x


class ConformerBlock(nn.Module):
    """
    Conformer block: 1/2 FFN -> Attention -> Conv -> 1/2 FFN -> LayerNorm
    All modules use residual connections.
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        conv_kernel_size: int = 31,
        pos_emb_init_len: int = 256,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.fc_factor = 0.5
        self.ffn1 = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=dropout_p,
        )
        self.ffn2 = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=dropout_p,
        )

        self.self_attn = MultiHeadedSelfAttentionModule(
            input_dim=encoder_dim,
            num_heads=num_attention_heads,
            max_len = pos_emb_init_len,
            dropout_p = dropout_p
        )

        self.conv_module = ConvolutionModule(
            in_channels=encoder_dim,
            kernel_size=conv_kernel_size,
            expansion_factor = conv_expansion_factor,
            dropout_p=dropout_p,
        )

        self.ln = nn.LayerNorm(encoder_dim)
        self.dropout = nn.Dropout(dropout_p)


    def forward(
        self,
        x: Tensor,
        pad_mask: Optional[Tensor] = None,
        att_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        x:        [B, T, encoder_dim]
        att_mask: [B, T, T] attention mask (True where padding)
        pad_mask: [B, T] padding mask (True where padding)
        """

        # 1) First 1/2 FFN (Macaron-style)
        x = x + self.fc_factor * self.ffn1(x)

        residual = x
        att_out = self.self_attn(
            x,
            att_mask,
        )
        x = self.dropout(att_out)
        x = residual + x

        # 3) Convolution module
        x = x + self.conv_module(x, pad_mask=pad_mask)

        # 4) Second 1/2 FFN
        x = x + self.fc_factor * self.ffn2(x)

        # 5) Final LayerNorm
        x = self.ln(x)

        return x

class ConformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 512,
        num_layers: int = 17,
        num_attention_heads: int = 8,
        conv_kernel_size: int = 31,
        dropout_p: int = 0.1,
        pos_emb_init_len: int = int(36/0.04),
    ):
        super().__init__()
        self.feat_in = input_dim
        self.encoder_dim = encoder_dim

        self.pre_encode = StridingSubsampling(out_channels=encoder_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(encoder_dim * 20, encoder_dim),    #[B, T', 20*512] -> [B, T', encoder_dim]
            nn.Dropout(p=dropout_p),
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = ConformerBlock(
                encoder_dim=encoder_dim,
                num_attention_heads=num_attention_heads,
                feed_forward_expansion_factor = 4,
                conv_expansion_factor = 2,
                conv_kernel_size=conv_kernel_size,
                pos_emb_init_len=pos_emb_init_len,
                dropout_p=dropout_p
            )
            self.layers.append(layer)

    def forward(self, audio_signal: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor]:
        outputs, out_lengths  = self.pre_encode(audio_signal, lengths)  # [B, T, 80] -> [B, T', enc_dim*20]
        B, T_prime, _ = outputs.shape
        pad_mask = None
        att_mask = None
        if B > 1:
            # Create padding masks (True where padding)
            t_ids = torch.arange(T_prime, device=outputs.device).unsqueeze(0) # [1, T']
            out_lengths_dev = out_lengths.to(outputs.device)
            pad_mask = t_ids>=out_lengths_dev.unsqueeze(1)
            att_mask = pad_mask.unsqueeze(1) | pad_mask.unsqueeze(2)   # [B, T', T']

        outputs = self.input_proj(outputs)

        for layer in self.layers:
            outputs = layer(
                x=outputs,
                att_mask=att_mask,
                pad_mask=pad_mask,
            )

        return outputs, out_lengths

class ConformerCTC (ConformerEncoder):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out = nn.Linear(self.encoder_dim, num_classes, bias=False)

    def forward(self, audio_signal: Tensor, length: Tensor) -> Tensor:
        audio_signal, out_length = super().forward(audio_signal, length)
        logits = self.out(audio_signal)
        return logits, out_length

