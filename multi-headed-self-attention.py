import torch
import torch.nn as nn
from torchtyping import TensorType


class MultiHeadedSelfAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)

        head_size = attention_dim // num_heads
        self.multi_head_attention = nn.ModuleList(
            [
                self.SingleHeadAttention(embedding_dim, head_size)
                for _ in range(num_heads)
            ]
        )

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        head_outputs = []  # B * T * head_size which is attention_dim // num_heads

        for single_head_attention in self.multi_head_attention:
            head_outputs.append(single_head_attention(embedded))

        # dim=2 because we want to make head_size dim attention_dim
        concatinated = torch.cat(head_outputs, dim=2)
        return torch.round(concatinated, decimals=4)

    class SingleHeadAttention(nn.Module):
        def __init__(self, embedding_dim: int, attention_dim: int):
            super().__init__()
            torch.manual_seed(0)
            self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            k = self.key_gen(embedded)
            q = self.query_gen(embedded)
            v = self.value_gen(embedded)

            scores = q @ torch.transpose(k, 1, 2)  # @ is the same as torch.matmul()
            context_length, attention_dim = k.shape[1], k.shape[2]
            scores = scores / (attention_dim**0.5)

            lower_triangular = torch.tril(torch.ones(context_length, context_length))
            mask = lower_triangular == 0
            scores = scores.masked_fill(mask, float("-inf"))
            scores = nn.functional.softmax(scores, dim=2)

            return scores @ v
