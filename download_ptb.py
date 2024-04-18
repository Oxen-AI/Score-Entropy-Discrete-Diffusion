from datasets import load_dataset

ds = load_dataset("ptb_text_only")

with open('data/ptb_text_only.txt', 'w') as f:
    for d in ds['train']:
        f.write(d['sentence'] + '\n')