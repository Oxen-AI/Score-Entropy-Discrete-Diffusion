from torch.utils.data import Dataset
from oxen.datasets import load_dataset, download, _load_hf
from datasets import load_dataset
import torch
import os
from tqdm import tqdm

class OpenSubtitlesDataset(Dataset):
    def __init__(self, tokenizer, train=True, num_examples=-1, seq_len=32, num_proc=8):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        # TODO: Allow for output destination for file
        if train:
            filename = "train_10M/open_subtitles.train.parquet"
        else:
            filename = "dev/open_subtitles.dev.parquet"
        if not os.path.exists(filename):
            download("datasets/BabyLM_2024", path=filename)
        
        self.dataset = _load_hf(filename)
        self.dataset = self.dataset['train']
        if num_examples > 0:
            print(f"Subsampling dataset to {num_examples} examples")
            self.dataset = self.dataset.select(range(num_examples))
        PAD = self.tokenizer.pad_token_id
        
        # Add sequential texts together with newlines
        data = []
        for i in tqdm(range(len(self.dataset)-1)):
            data.append({"text": self.dataset[i]['text'] + "\n" + self.dataset[i+1]['text']})
        
        def preprocess_and_tokenize(example):
            global max_len
            text = example["text"]
            tokens = tokenizer(text, return_attention_mask=False)
            # Pad batch to block_size
            for i in range(len(tokens['input_ids'])):
                if len(tokens['input_ids'][i]) < seq_len:
                    tokens['input_ids'][i] = tokens['input_ids'][i] + [PAD] * (seq_len - len(tokens['input_ids'][i]))
                else:
                    tokens['input_ids'][i] = tokens['input_ids'][i][:seq_len]

            return tokens
        
        print("Preprocessing and tokenizing dataset")
        self.dataset = data.map(preprocess_and_tokenize, batched=True, num_proc=num_proc)
        print(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return torch.Tensor(self.dataset[idx]['input_ids']).long()