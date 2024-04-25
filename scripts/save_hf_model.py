import torch
import argparse

from load_model import load_model

def main():
    parser = argparse.ArgumentParser(description="Save model to disk")
    parser.add_argument("--model", default="louaaron/sedd-medium", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    print("Loading model...")
    model, _, _ = load_model(args.model, device='cuda')

    print("Saving model...")
    torch.save(model.state_dict(), args.output)

if __name__=="__main__":
    main()