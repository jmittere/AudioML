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
        n_mels,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        max_len=2000  # for positional encoding

    ):
        super(MelTransformer, self).__init__()

        self.input_proj = nn.Linear(1, d_model)

        #TODO: change max_lemb_emb, T*F = ~26000
        max_len_emb = 26000
        #TODO: Implement specific frequency and time encodings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len_emb, d_model))

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

        self.output_proj = nn.Linear(d_model, 1)
    

    def forward(self, x):
        # x: (B, L, 1)

        x = self.input_proj(x)  # (B, L, D)

        L = x.size(1)
        x = x + self.pos_embedding[:, :L, :]

        mask = generate_causal_mask(L, x.device)

        x = self.transformer(x, mask=mask)

        x = self.output_proj(x)  # (B, L, 1)

        return x