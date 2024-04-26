
import argparse
from transformers import GPT2TokenizerFast
import torch
import os
import yaml

from sedd.models.sedd import SEDD
from sedd.models.graph import AbsorbingGraph
from sedd.models.noise import LogLinearNoise
from sedd.trainer.loss import loss_fn

def main():
    parser = argparse.ArgumentParser(description="Check the loss function")
    parser.add_argument("text", type=str)
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-t", "--timestep", type=float, default=0.01)

    args = parser.parse_args()
    device = torch.device('cuda')
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    print(f"Got text:\n{args.text}\n")
    tokens = tokenizer(args.text, return_tensors='pt')['input_ids'].to(device)
    print(f"Got tokens:\n{tokens}\n")

    # Demo how to sample noise at different timesteps
    print(f"Noise by timestep {args.timestep}...")
    
    graph = AbsorbingGraph(tokenizer.vocab_size)
    noise = LogLinearNoise()
    
    # Load the model onto GPU
    cfg = os.path.join(args.model, 'config.yaml')
    with open(cfg, 'r') as f:
        cfg = yaml.full_load(f)

    model = SEDD(cfg, tokenizer.vocab_size).to(device)
    model_file = os.path.join(args.model, "checkpoint.pth")
    loaded_state = torch.load(model_file, map_location=device)
    model.load_state_dict(loaded_state)
    
    # Sample noise at timestep
    t = (torch.ones(1) * args.timestep).to(device)
    sigma, _ = noise(t)

    print(f"Got noise: {sigma.item()}\n")
    print(sigma[:, None])
    
    # Noisify the tokens
    loss = loss_fn(tokens, model, noise.to(device), graph, train=False, t=t)

    print(f"Got loss: {loss.item()}\n")

if __name__=="__main__":
    main()