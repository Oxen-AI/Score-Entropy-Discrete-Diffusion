from torch.utils.data import Dataset
import torch
import math

class BrownCowDataset(Dataset):
    def __init__(self, tokenizer, num_examples=128, seq_len=32):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.num_examples = num_examples
        self.data = [
            "Ow now brown cow",
            "How now brown cow",
            "Wow now brown cow",
            "Unique New York",
            "Oxen are a type of cow",
        ]
        self.tokens = torch.Tensor(tokenizer(self.data, padding='max_length', max_length=self.seq_len)['input_ids']).long()
        print("Tokens shape: ", self.tokens.shape)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        idx = idx % len(self.tokens)
        return self.tokens[idx]