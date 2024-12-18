import torch
import torch.nn as nn

class TabularTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, dim, depth, heads, mlp_dim):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, dim)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=0.1
            ),
            num_layers=depth
        )
        
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, num_classes)
        )
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x)
        x = x.squeeze(1)  # Remove sequence dimension
        return self.mlp_head(x) 