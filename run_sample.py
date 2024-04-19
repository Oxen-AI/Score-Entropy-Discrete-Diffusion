import torch
import argparse

from load_model import load_model
from transformers import GPT2TokenizerFast
import torch.nn.functional as F
import sampling


def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="louaaron/sedd-medium", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--print_intermediate", type=bool, default=False)
    parser.add_argument("--print_sequential", type=bool, default=False)
    args = parser.parse_args()

    
    device = torch.device('cuda')
    model, graph, noise = load_model(args.model_path, device)
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    if args.print_intermediate:
        sampling_fn = sampling.get_pc_sampler(
            graph,
            noise,
            (args.batch_size, 1024),
            'analytic',
            args.steps,
            device=device,
            tokenizer=tokenizer,
            should_print_sequential=args.print_sequential
        )
    else:
        sampling_fn = sampling.get_pc_sampler(
            graph, noise, (args.batch_size, 1024), 'analytic', args.steps, device=device
        )

    samples = sampling_fn(model)

    text_samples = tokenizer.batch_decode(samples)
    for i in text_samples:
        print(i)
        print("=================================================")

if __name__=="__main__":
    main()