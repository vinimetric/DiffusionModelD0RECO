import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, time_emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.SiLU(),
            nn.LayerNorm(out_features),
            nn.Linear(out_features, out_features)
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_features)
        )
        
        self.residual_layer = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x, t):
        # x: (batch_size, in_features)
        # t: (batch_size, time_emb_dim)
        time_emb = self.time_mlp(t)
        h = self.mlp(x) + time_emb
        return h + self.residual_layer(x)

class DDPM_MLP(nn.Module):
    def __init__(self, feature_dim=8, hidden_dim=256, time_emb_dim=128, num_layers=4):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        
        self.input_layer = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU()
        )
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, time_emb_dim) for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, x, time):
        # x: (batch_size, feature_dim)
        # time: (batch_size,) - 1D tensor of timesteps
        t = self.time_mlp(time)
        x = self.input_layer(x)
        
        for block in self.res_blocks:
            x = block(x, t)
            
        return self.output_layer(x)
