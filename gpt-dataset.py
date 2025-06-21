import torch
from typing import List, Tuple


class Solution:
    def batch_loader(
        self, raw_dataset: str, context_length: int, batch_size: int
    ) -> Tuple[List[List[str]]]:
        torch.manual_seed(0)

        tokens = raw_dataset.split()
        indices = torch.randint(
            low=0, high=len(tokens) - context_length, size=(batch_size,)
        ).tolist()

        X = []
        Y = []

        for i in indices:
            X.append(tokens[i : i + context_length])
            Y.append(tokens[i + 1 : i + 1 + context_length])

        return X, Y
