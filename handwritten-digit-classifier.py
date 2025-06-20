import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)

        self.first_layer = nn.Linear(in_features=28 * 28, out_features=512)
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.final_layer = nn.Linear(in_features=512, out_features=10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)

        output = self.sigmoid(
            self.final_layer(self.dropout(self.ReLU(self.first_layer(images))))
        )
        return torch.round(output, decimals=4)
