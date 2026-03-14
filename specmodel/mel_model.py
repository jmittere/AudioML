import torch.nn as nn

class MelTransformer(nn.Module):
    def __init__(
        self,
        n_mels=80,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1
    ):
        super(MelTransformer, self).__init__()

        self.input_proj = nn.Linear(n_mels, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
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
        """
        x: (B, T, 80)
        """
        x = self.input_proj(x)          # (B, T, D)
        x = self.transformer(x)         # (B, T, D)
        x = self.output_proj(x)         # (B, T, 80)
        return x