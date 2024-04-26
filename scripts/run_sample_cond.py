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
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--show_intermediate", action='store_true')
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    
    # Config should be saved in the model directory
    cfg = os.path.join(args.model, 'config.yaml')
    with open(cfg, 'r') as f:
        cfg = yaml.full_load(f)

    # Load the tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    print("Vocab size: ", tokenizer.vocab_size)

    print("Complete prefix: ", args.prefix)
    print("with suffix: ", args.suffix)

    prefix_ids = tokenizer(args.prefix).input_ids
    suffix_ids = tokenizer(args.suffix).input_ids
    input_ids = prefix_ids + suffix_ids
    input_locs = list(range(len(prefix_ids))) + list(range(1024-len(suffix_ids), 1024))

    # more generaly commands can be defined with something like below:
    # input_ids = [0, 1, 512, 8080, 50256, 20000]
    # input_locs = [5, 6, 19, 20, 1000, 10001]

    input_ids = torch.tensor(input_ids, device="cuda")[None].repeat(args.batch_size, 1)

    def proj_fun(x):
        x[:, input_locs] = input_ids
        return x

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
    texts = sampler.sample(tokenizer, model, graph, noise, steps=args.steps, show_intermediate=args.show_intermediate, projector=proj_fun)

    for i in texts:
        print("="*80)
        print(i)
        print("="*80)

if __name__=="__main__":
    main()
