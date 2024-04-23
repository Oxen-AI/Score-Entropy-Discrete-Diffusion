from torch.utils.data import Dataset
from datasets import load_dataset
import torch

class Wikitext2Dataset(Dataset):
    def __init__(self, tokenizer, num_examples, cache_dir="./data", seq_len=32, num_proc=8):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.num_examples = num_examples
        
        self.dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", cache_dir=cache_dir)
        data = self.dataset['train']
        
        PAD = self.tokenizer.pad_token_id
        
        def preprocess_and_tokenize(example):
            text = example["text"]
            tokens = tokenizer(text, return_attention_mask=False)
            # Pad batch to block_size
            for i in range(len(tokens['input_ids'])):
                if len(tokens['input_ids'][i]) < seq_len:
                    tokens['input_ids'][i] = tokens['input_ids'][i] + [PAD] * (seq_len - len(tokens['input_ids'][i]))
                else:
                    tokens['input_ids'][i] = tokens['input_ids'][i][:seq_len]

            return tokens
        
        self.dataset = data.map(preprocess_and_tokenize, batched=True, num_proc=num_proc, load_from_cache_file=True)
        print(self.dataset)
        # self.dataset = tokenized_dataset.with_format('torch')

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        idx = idx % len(self.dataset)
        return torch.Tensor(self.dataset[idx]['input_ids']).long()