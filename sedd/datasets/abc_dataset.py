from torch.utils.data import Dataset
import torch

class ABCDataset(Dataset):
    def __init__(self, tokenizer, num_examples=128, seq_len=32):
        self.seq_len = seq_len
        self.num_examples = num_examples
        self.data = "abcdefghijklmnopqrstuvwxyz" + (tokenizer.pad_token * (seq_len - 26))
        self.tokens = torch.Tensor(tokenizer(self.data)['input_ids']).long()

    def __len__(self):
        return self.num_examples

    def __getitem__(self, _idx):
        return self.tokens