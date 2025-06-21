import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        self.embedding_layer = nn.Embedding(
            num_embeddings=vocabulary_size, embedding_dim=16
        )
        self.linear_layer = nn.Linear(in_features=16, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        output = self.sigmoid(
            self.linear_layer(torch.mean(self.embedding_layer(x), dim=1))
        )
        return torch.round(output, decimals=4)
