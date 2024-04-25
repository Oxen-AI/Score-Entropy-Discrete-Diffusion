
import argparse
from transformers import GPT2TokenizerFast

def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("text", type=str)
    parser.add_argument("-n", "--num_shift", type=int, default=1)

    args = parser.parse_args()
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    print(f"Got text:\n{args.text}\n")
    tokens = tokenizer(args.text, return_tensors='pt')['input_ids']
    print(f"Got tokens:\n{tokens}\n")
    
    print(f"Shift by {args.num_shift}...")
    shifted_tokens = (tokens + args.num_shift) % tokenizer.vocab_size
    print(f"Got shifted tokens:\n{shifted_tokens}\n")
    
    # TODO: Sample with sampler at different times and like training
    
    # Decode the shifted tokens
    shifted_text = tokenizer.decode(shifted_tokens[0])
    print(f"Shifted text:\n{shifted_text}")

if __name__=="__main__":
    main()