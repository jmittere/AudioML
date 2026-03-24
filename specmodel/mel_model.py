import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def generate_causal_mask(T, device):
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1)
        return mask.bool()

class MelTransformer(nn.Module):
    def __init__(
        self,
        n_mels=80,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        max_len=2000  # for positional encoding

    ):
        super(MelTransformer, self).__init__()

        self.input_proj = nn.Linear(n_mels, d_model)

        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            activation = "relu",
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.output_proj = nn.Linear(d_model, n_mels)
    

    def forward(self, x):
        # x: (B, T, 80)

        #input projection to encoder
        x = self.input_proj(x)     # (B, T, D)

        #positional encoding
        T = x.size(1)
        x = x + self.pos_embedding[:, :T, :]

        #causal mask
        causal_mask = generate_causal_mask(T, x.device)

        x = self.transformer(x, mask=causal_mask)

        x = self.output_proj(x)    # (B, T, 80)
    
        return x