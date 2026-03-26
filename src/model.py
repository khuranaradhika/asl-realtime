"""
src/model.py

SignTransformer — compact keypoint-based temporal transformer for ASL recognition.

Input:  (B, T, 126) — batch of keypoint sequences
Output: (T, B, n_classes + 1) — log-softmax logits for CTC loss
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding — injects temporal position information
    into the frame embeddings before the transformer encoder.
    """
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return self.dropout(x + self.pe[:, :x.size(1)])


class SignTransformer(nn.Module):
    """
    Compact transformer encoder for isolated ASL sign recognition.

    Architecture:
        Linear projection (126 → d_model)
        → Positional encoding
        → N × TransformerEncoderLayer (self-attention + FFN)
        → Linear classifier (d_model → n_classes + 1)

    The output is shaped (T, B, C) for PyTorch's nn.CTCLoss.

    Args:
        d_model:        feature dimension (default 128 for student, 512 for teacher)
        nhead:          number of attention heads
        n_layers:       number of transformer encoder layers
        dim_feedforward: FFN hidden dimension
        n_classes:      vocabulary size (number of ASL signs)
        dropout:        dropout rate
        input_dim:      keypoint feature dimension (126 for hands-only)
    """

    def __init__(
        self,
        d_model:        int = 128,
        nhead:          int = 4,
        n_layers:       int = 3,
        dim_feedforward:int = 256,
        n_classes:      int = 100,
        dropout:        float = 0.1,
        input_dim:      int = 126,
    ):
        super().__init__()
        self.d_model   = d_model
        self.n_classes = n_classes

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = nhead,
            dim_feedforward= dim_feedforward,
            dropout        = dropout,
            batch_first    = True,   # expects (B, T, C)
            norm_first     = True,   # pre-norm — more stable training
        )
        self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, n_classes + 1)  # +1 for CTC blank token

        self._init_weights()

    def _init_weights(self):
        """Xavier init for projection layers."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                    (B, T, 126) keypoint sequences
            src_key_padding_mask: (B, T) bool mask — True = padded frame to ignore

        Returns:
            log_probs: (T, B, n_classes + 1) — log-softmax for CTCLoss
        """
        x = self.input_proj(x)                              # (B, T, d_model)
        x = self.pos_enc(x)                                 # (B, T, d_model)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, T, d_model)
        x = self.classifier(x)                              # (B, T, C)
        x = x.permute(1, 0, 2)                              # (T, B, C) for CTCLoss
        return F.log_softmax(x, dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_student_model(n_classes: int = 100) -> SignTransformer:
    """CPU-deployable student model (~1.2M params, <15ms latency)."""
    return SignTransformer(
        d_model=128, nhead=4, n_layers=3,
        dim_feedforward=256, n_classes=n_classes, dropout=0.1)


def build_teacher_model(n_classes: int = 100) -> SignTransformer:
    """Larger teacher model (~18M params, GPU only)."""
    return SignTransformer(
        d_model=512, nhead=8, n_layers=6,
        dim_feedforward=1024, n_classes=n_classes, dropout=0.1)


def make_padding_mask(input_lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Create a boolean padding mask for the transformer.
    True = padded (ignore), False = real data.

    Args:
        input_lengths: (B,) actual sequence lengths
        max_len:       padded sequence length T

    Returns:
        mask: (B, T) bool tensor
    """
    batch_size = input_lengths.size(0)
    mask = torch.arange(max_len, device=input_lengths.device).unsqueeze(0)
    mask = mask.expand(batch_size, -1) >= input_lengths.unsqueeze(1)
    return mask  # True where padded


if __name__ == "__main__":
    # Quick sanity check
    model = build_student_model(n_classes=100)
    print(f"Student parameters: {model.count_parameters():,}")

    x    = torch.randn(4, 80, 126)   # (B=4, T=80, keypoints=126)
    lens = torch.tensor([80, 60, 45, 30])
    mask = make_padding_mask(lens, max_len=80)
    out  = model(x, src_key_padding_mask=mask)
    print(f"Output shape: {out.shape}")  # expected: (80, 4, 101)
