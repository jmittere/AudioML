import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def generate_causal_mask(T, device):
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1)
        return mask.bool()

###-- Used for Frame*Freq_bin auto regression --###
class MelTransformerFrameBin(nn.Module):
    def __init__(
        self,
        n_mels,
        max_time_frames,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dim_feedforward=512,
        dropout=0.1
    ):
        self.n_mels = n_mels
        self.max_time_frames = max_time_frames

        super(MelTransformerFrameBin, self).__init__()

        self.input_proj = nn.Linear(1, d_model)

        self.time_embed = nn.Embedding(self.max_time_frames, d_model)
        self.freq_embed = nn.Embedding(self.n_mels, d_model)

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

        B, L, D = x.shape
        F = self.n_mels

        T = (L + F - 1) // F  # ceil division

        time_indices = torch.arange(T, device=x.device).unsqueeze(1).expand(T, F).reshape(-1) #[0,0,0,..., 1,1,1,..., 2,2,2,...]
        freq_indices = torch.arange(F, device=x.device).unsqueeze(0).expand(T, F).reshape(-1)

        time_indices = time_indices[:L]
        freq_indices = freq_indices[:L]

        pos = self.time_embed(time_indices) + self.freq_embed(freq_indices)
        pos = pos.unsqueeze(0)
        x = x + pos

        mask = generate_causal_mask(L, x.device)

        x = self.transformer(x, mask=mask)

        x = self.output_proj(x)  # (B, L, 1)

        return x

###-- Used for Frame auto regression --###
class MelTransformerFrame(nn.Module):
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
        super(MelTransformerFrame, self).__init__()

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
    
###-- Used for Frame auto regression but predicts deltas between frames--###
class MelTransformerFrameDelta(nn.Module):
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
        super(MelTransformerFrameDelta, self).__init__()

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

        #delta between frames
        delta = self.output_proj(x)    # (B, T, 80)
    
        return delta