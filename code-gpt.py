import torch
import torch.nn as nn
from torchtyping import TensorType


#  Remember to include an additional LayerNorm after the block sequence and before the final linear layer
class GPT(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        model_dim: int,
        num_blocks: int,
        num_heads: int,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.word_embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=model_dim
        )
        self.position_embeddings = nn.Embedding(
            num_embeddings=context_length, embedding_dim=model_dim
        )
        self.transformer_blocks = nn.Sequential()
        for _ in range(num_blocks):
            self.transformer_blocks.append(
                self.TransformerBlock(model_dim=model_dim, num_heads=num_heads)
            )
        self.final_layer_norm = nn.LayerNorm(model_dim)
        self.vocabulary_projection = nn.Linear(
            in_features=model_dim, out_features=vocab_size
        )

    def forward(self, context: TensorType[int]) -> TensorType[float]:
        torch.manual_seed(0)

        embedded = self.word_embeddings(context)
        context_length = context.shape[1]
        positions = torch.arange(context_length)
        positional_encoded = self.position_embeddings(positions)

        embedded = embedded + positional_encoded

        transformered = self.transformer_blocks(embedded)
        final_normed = self.final_layer_norm(transformered)
        projected = self.vocabulary_projection(final_normed)

        probabilities = nn.functional.softmax(projected, dim=2)

        return torch.round(probabilities, decimals=4)

    # Do NOT modify the code below this line
    class TransformerBlock(nn.Module):

        class MultiHeadedSelfAttention(nn.Module):

            class SingleHeadAttention(nn.Module):
                def __init__(self, model_dim: int, head_size: int):
                    super().__init__()
                    torch.manual_seed(0)
                    self.key_gen = nn.Linear(model_dim, head_size, bias=False)
                    self.query_gen = nn.Linear(model_dim, head_size, bias=False)
                    self.value_gen = nn.Linear(model_dim, head_size, bias=False)

                def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                    k = self.key_gen(embedded)
                    q = self.query_gen(embedded)
                    v = self.value_gen(embedded)

                    scores = q @ torch.transpose(
                        k, 1, 2
                    )  # @ is the same as torch.matmul()
                    context_length, attention_dim = k.shape[1], k.shape[2]
                    scores = scores / (attention_dim**0.5)

                    lower_triangular = torch.tril(
                        torch.ones(context_length, context_length)
                    )
                    mask = lower_triangular == 0
                    scores = scores.masked_fill(mask, float("-inf"))
                    scores = nn.functional.softmax(scores, dim=2)

                    return scores @ v

            def __init__(self, model_dim: int, num_heads: int):
                super().__init__()
                torch.manual_seed(0)
                self.att_heads = nn.ModuleList()
                for i in range(num_heads):
                    self.att_heads.append(
                        self.SingleHeadAttention(model_dim, model_dim // num_heads)
                    )

            def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                head_outputs = []
                for head in self.att_heads:
                    head_outputs.append(head(embedded))
                concatenated = torch.cat(head_outputs, dim=2)
                return concatenated

        class VanillaNeuralNetwork(nn.Module):

            def __init__(self, model_dim: int):
                super().__init__()
                torch.manual_seed(0)
                self.up_projection = nn.Linear(model_dim, model_dim * 4)
                self.relu = nn.ReLU()
                self.down_projection = nn.Linear(model_dim * 4, model_dim)
                self.dropout = nn.Dropout(0.2)  # using p = 0.2

            def forward(self, x: TensorType[float]) -> TensorType[float]:
                torch.manual_seed(0)
                return self.dropout(
                    self.down_projection(self.relu(self.up_projection(x)))
                )

        def __init__(self, model_dim: int, num_heads: int):
            super().__init__()
            torch.manual_seed(0)
            self.attention = self.MultiHeadedSelfAttention(model_dim, num_heads)
            self.linear_network = self.VanillaNeuralNetwork(model_dim)
            self.first_norm = nn.LayerNorm(model_dim)
            self.second_norm = nn.LayerNorm(model_dim)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            torch.manual_seed(0)
            embedded = embedded + self.attention(
                self.first_norm(embedded)
            )  # skip connection
            embedded = embedded + self.linear_network(
                self.second_norm(embedded)
            )  # another skip connection
            return embedded
