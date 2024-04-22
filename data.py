import re
import string
from transformers import GPT2TokenizerFast, AutoTokenizer
from tokenizers import Tokenizer
from transformers import CanineTokenizer
from datasets import load_dataset
from itertools import chain
import numpy as np
import torch
import string
from character_tokenizer import CharacterTokenizer


import urllib.request
import zipfile
import requests
import json
from datasets import Dataset

from torch.utils.data import DataLoader


def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data


def wt_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string

def ptb_detokenizer(x):
    x = x.replace(" 's", "'s")
    x = x.replace("s ' ", "s' ")
    x = x.replace(" n't", "n't")
    x = x.replace(" \n ", "\n")
    x = x.replace("\\/", "/")
    for _ in range(10):
        x = x.replace(" N ", " 1 ")
    x = x.replace("$ 1", "$1")
    x = x.replace("# 1", "#1")
    x = x.replace("<unk>", "?")
    return x

def lm1b_detokenizer(x):
    x = x.replace('http : / / ', 'http://')
    x = x.replace('https : / / ', 'https://')
    x = re.sub(r' \'(\w+)', r"'\1", x)
    x = re.sub(r' (\w+) \. ', r' \1. ', x)
    x = re.sub(r' (\w+) \.$', r' \1.', x)
    x = x.replace(' ? ', '? ')
    x = re.sub(r' \?$', '?', x)
    x = x.replace(' ! ', '! ')
    x = re.sub(r' \!$', '!', x)
    x = x.replace(' , ', ', ')
    x = x.replace(' : ', ': ')
    x = x.replace(' ; ', '; ')
    x = x.replace(' / ', '/')
    x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
    x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
    x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
    x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
    x = x.replace('$ ', '$')
    x = x.replace('£ ', '£')
    return x


def lambada_detokenizer(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return '\n'+text.strip()

def read_jsonl_to_list(url):
    response = requests.get(url, stream=True)
    data_list = []

    # Process each line in the response content
    for line in response.iter_lines(decode_unicode=True):
        if line:
            data = json.loads(line)
            data_list.append(data)

    return data_list

def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    

    lambada_data = read_jsonl_to_list(url)
    dataset = Dataset.from_list(lambada_data)
    return dataset

def get_baby_names_dataset(name):
    url = f"https://hub.oxen.ai/api/repos/datasets/baby_names/file/main/{name}.jsonl"
    data = read_jsonl_to_list(url)
    dataset = Dataset.from_list(data)
    return dataset


def get_dataset(name, mode, cache_dir=None, block_size=1024, num_proc=8):
    if name == "wikitext103":
        dataset = load_dataset("wikitext", name="wikitext-103-raw-v1", cache_dir=cache_dir)
    elif name == "wikitext2":
        dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", cache_dir=cache_dir)
    elif name == "ptb":
        dataset = load_dataset("ptb_text_only", cache_dir=cache_dir)
    elif name == "lambada":
        dataset = get_lambada_test_dataset()
    elif name == "baby_names_train":
        dataset = get_baby_names_dataset('train')
    elif name == "baby_names_test":
        dataset = get_baby_names_dataset('test')
    else:
        dataset = load_dataset(name, cache_dir=cache_dir)

    if name == "lambada" or name.startswith("baby_names"):
        data = dataset
    else:
        data = dataset[mode]

    if name.startswith("wikitext"):
        detokenizer = wt_detokenizer
    elif name == "ptb":
        detokenizer = ptb_detokenizer
    elif name == "lm1b":
        detokenizer = lm1b_detokenizer
    elif name == "lambada":
        detokenizer = lambada_detokenizer
    else:
        detokenizer = None

    def _apply_detokenizer(detokenizer):
        def detok(text):
            for i, t in enumerate(text, 0):
                 text[i] = detokenizer(t)
            return text
        return detok

    # tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    # tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    model_max_length = 2048
    tokenizer = CharacterTokenizer(model_max_length)

    print(len(tokenizer.get_vocab()))
    print(tokenizer.get_vocab())
    print(tokenizer.eos_token)
    EOS = tokenizer.encode(tokenizer.eos_token)[0]

    def preprocess_and_tokenize(example):
        text = example["text"]
        text = text[:model_max_length]
        tokens = tokenizer(text, return_attention_mask=False)
        # Pad batch to block_size
        for i in range(len(tokens['input_ids'])):
            if len(tokens['input_ids'][i]) < block_size:
                tokens['input_ids'][i] = tokens['input_ids'][i] + [EOS] * (block_size - len(tokens['input_ids'][i]))
            else:
                tokens['input_ids'][i] = tokens['input_ids'][i][:block_size]

        return tokens
    
    # filter out short strings
    data = data.filter(lambda x: len(x['text']) > 10)
    # tokenize the dataset
    tokenized_dataset = data.map(preprocess_and_tokenize, batched=True, num_proc=num_proc, load_from_cache_file=True)
    if name == "ptb":
        tokenized_dataset = tokenized_dataset.remove_columns('sentence')
    elif not name.startswith("baby_names"):
        tokenized_dataset = tokenized_dataset.remove_columns('text')
    

    # def group_texts(examples):
    #     # Concatenate all texts.
    #     concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    #     total_length = len(concatenated_examples[list(examples.keys())[0]])
    #     # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    #     # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    #     total_length = (total_length // block_size) * block_size
    #     # Split by chunks of max_len.
    #     result = {
    #         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
    #         for k, t in concatenated_examples.items()
    #     }
    #     return result

    # chunked_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=num_proc, load_from_cache_file=True)
    
    # for e in tokenized_dataset:
    #     print(e)
    #     print(tokenized_dataset[e])
    #     print()

    # def pad_texts(examples):
    #     for k in examples.keys():
    #         print(k)
    #         print(examples[k])
    #         for i in range(len(examples[k])):
    #             print(i)
    #             print(examples[k][i])
    #             examples[k][i] = [examples[k][i]] + [EOS] * (block_size - len(examples[k][i]))
    #         # examples[k] = [e + [EOS] * (block_size - len(e)) for e in examples[k]]
    #     return examples

    # padded_dataset = tokenized_dataset.map(tokenized_dataset, batched=True, num_proc=num_proc, load_from_cache_file=True)
    padded_dataset = tokenized_dataset.with_format('torch')

    return padded_dataset


def get_dataloaders(config):
    train_set = get_dataset(config['data']['train'], "train", cache_dir=config['data']['cache_dir'], block_size=config['model']['length'])
    valid_set = get_dataset(config['data']['valid'], "validation" if config['data']['valid'] != "text8" else "test", cache_dir=config['data']['cache_dir'], block_size=config['model']['length'])

    print(train_set)
    for i in range(5):
        print(i)
        print(train_set[i])
        print()

    train_sampler = None
    test_sampler = None

    train_loader = cycle_loader(DataLoader(
        train_set,
        shuffle=(train_sampler is None),
    ))
    valid_loader = cycle_loader(DataLoader(
        valid_set,
        shuffle=(test_sampler is None),
    ))
    return train_loader, valid_loader

