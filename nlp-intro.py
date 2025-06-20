import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution:
    def get_dataset(
        self, positive: List[str], negative: List[str]
    ) -> TensorType[float]:
        word_set = set()
        all_texts = positive + negative

        for text in all_texts:
            for word in text.split():
                word_set.add(word)

        sorted_word_list = sorted(word_set)
        word_to_encoded = {}
        for i, word in enumerate(sorted_word_list):
            word_to_encoded[word] = i + 1

        tensors = []
        for text in all_texts:
            curr = []
            for word in text.split():
                curr.append(word_to_encoded[word])
            tensors.append(torch.tensor(curr))

        return nn.utils.rnn.pad_sequence(tensors, batch_first=True)
