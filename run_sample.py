import os
import torch
import argparse
import yaml
from transformers import GPT2TokenizerFast

from sedd.models.sedd import SEDD
from sedd.models.graph import AbsorbingGraph
from sedd.models.noise import LogLinearNoise
from sedd.models.sampler import Sampler

def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", default="gpt2", type=str)
    parser.add_argument("--show_intermediate", action='store_true')
    parser.add_argument("--steps", type=int, default=1024)
    args = parser.parse_args()
    
    # Config should be saved in the model directory
    cfg = os.path.join(args.model, 'config.yaml')
    with open(cfg, 'r') as f:
        cfg = yaml.full_load(f)

    # Load the tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    # Load the model onto GPU
    device = torch.device('cuda')
    model = SEDD(cfg, tokenizer.vocab_size).to(device)
    model_file = os.path.join(args.model, "checkpoint.pth")
    loaded_state = torch.load(model_file, map_location=device)
    model.load_state_dict(loaded_state)

    # Load the transition graph
    graph = AbsorbingGraph(tokenizer.vocab_size)
    
    # Load the noise function
    noise = LogLinearNoise().to(device)

    sampler = Sampler(cfg, device=device)
    texts = sampler.sample(tokenizer, model, graph, noise, steps=args.steps, show_intermediate=args.show_intermediate)

    for i in texts:
        print("="*80)
        print(i)
        print("="*80)

if __name__=="__main__":
    main()