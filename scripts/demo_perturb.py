
import argparse
from transformers import GPT2TokenizerFast
import torch

from sedd.models.graph import AbsorbingGraph
from sedd.models.noise import LogLinearNoise

def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("text", type=str)
    parser.add_argument("-t", "--timestep", type=float, default=0.01)

    args = parser.parse_args()
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    print(f"Got text:\n{args.text}\n")
    tokens = tokenizer(args.text, return_tensors='pt')['input_ids']
    print(f"Got tokens:\n{tokens}\n")

    # Demo how to sample noise at different timesteps
    print(f"Noise by timestep {args.timestep}...")
    
    graph = AbsorbingGraph(tokenizer.vocab_size)
    noise = LogLinearNoise()
    
    # Sample noise at timestep
    t = torch.ones(1) * args.timestep
    sigma, _ = noise(t)

    print(f"Got noise: {sigma.item()}\n")
    print(sigma[:, None])
    
    perturbed_tokens = graph.sample_transition(tokens, sigma[:, None])

    print(f"Got shifted tokens:\n{perturbed_tokens}\n")
    
    
    # Decode the shifted tokens
    shifted_text = tokenizer.decode(perturbed_tokens[0])
    print(f"Shifted text:\n{shifted_text}")

if __name__=="__main__":
    main()