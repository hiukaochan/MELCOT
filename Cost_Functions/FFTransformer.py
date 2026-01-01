import torch
import torch.nn as nn

class FFTransformerModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_countries: int,
        n_sports: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):

        super(FFTransformerModel, self).__init__()
        self.input_proj = nn.Linear(n_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(d_model, n_sports)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == self.input_proj.in_features:
            x = x.permute(1, 0)


        x = self.input_proj(x)             

        x = x.unsqueeze(0)
        x = self.transformer(x)            

        x = x.squeeze(0)                   
        out = self.output_layer(x)         
        return out.permute(1, 0)
