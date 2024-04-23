
import os
import os.path
import yaml
import oxen

import torch
import utils
import yaml


import argparse
from sedd.datasets.ox_dataset import OxDataset
from sedd.datasets.brown_cow_dataset import BrownCowDataset
from sedd.datasets.wikitext2_dataset import Wikitext2Dataset
from sedd.datasets.open_subtitles_dataset import OpenSubtitlesDataset
from sedd.datasets.abc_dataset import ABCDataset

from sedd.tokenizers.ox_tokenizer import OxTokenizer
from sedd.tokenizers.abc_tokenizer import ABCTokenizer
from sedd.models.noise import LogLinearNoise
# from sedd.models.simple_sedd import SEDD
from sedd.models.sedd import SEDD
from sedd.models.sampler import Sampler
from sedd.models.graph import AbsorbingGraph
from sedd.trainer.trainer import Trainer
from sedd.eval.evaluator import Evaluator

from aim import Run

# from sedd.models.simple_sedd import SEDD
from torch.utils.data import DataLoader

def print_devices(device):
    if torch.cuda.is_available():
        print("Found {} CUDA devices.".format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(
                "{} \t Memory: {:.2f}GB".format(
                    props.name, props.total_memory / (1024 ** 3)
                )
            )
    else:
        print("WARNING: Using device {}".format(device))
    print(f"Using device: {device}")
    print(f"Found {os.cpu_count()} total number of CPUs.")

def main():
    args = argparse.ArgumentParser(description="Train SEDD")
    args.add_argument("--config", type=str, default="configs/config.yaml")
    args.add_argument("--output", type=str, default="output")
    args.add_argument("--repo", type=str, default="ox/SEDD_dev")
    args = args.parse_args()
    
    # load in tokenizer
    # tokenizer = OxTokenizer()
    tokenizer = ABCTokenizer()

    with open(args.config, 'r') as f:
        cfg = yaml.full_load(f)

    cfg['tokens'] = tokenizer.vocab_size
    cfg['data'] = {}
    cfg['data']['remote_repo'] = args.repo
    cfg['training']['output_dir'] = args.output

    print(cfg)

    work_dir = cfg['training']['output_dir']

    # Create directories for experimental logs
    sample_dir = os.path.join(work_dir, "samples")
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    utils.makedirs(sample_dir)
    utils.makedirs(checkpoint_dir)
    utils.makedirs(os.path.dirname(checkpoint_meta_dir))

    print(work_dir)
    print(cfg)

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print_devices(device)

    # Create remote oxen repo
    repo = oxen.RemoteRepo(cfg['data']['remote_repo'])
    if not repo.exists():
        repo.create()

    # Save config file for this run
    repo.add('configs/config.yaml')
    repo.commit("Added config file")

    # build token graph
    graph = AbsorbingGraph(tokenizer.vocab_size)

    # build score model
    score_model = SEDD(cfg).to(device)

    num_parameters = sum(p.numel() for p in score_model.parameters())
    print(f"Number of parameters in the model: {num_parameters}")

    # train_ds = DataLoader(OpenSubtitlesDataset(tokenizer, seq_len=cfg['model']['length']), batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=4)
    # eval_ds = DataLoader(OpenSubtitlesDataset(tokenizer, seq_len=cfg['model']['length'], num_examples=128))

    train_ds = DataLoader(ABCDataset(tokenizer, seq_len=cfg['model']['length'], num_examples=10000), batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=4)
    eval_ds = DataLoader(ABCDataset(tokenizer, seq_len=cfg['model']['length'], num_examples=128))

    noise = LogLinearNoise().to(device)

    run = Run()
    run["hparams"] = cfg
    
    def eval(state):
        evaluator = Evaluator(eval_ds, run, cfg, device=device)
        return evaluator.evaluate(state)
    
    def sample(state):
        sampler = Sampler(tokenizer, sample_dir, cfg)
        return sampler.sample(state)
        
    trainer = Trainer(
        run,
        score_model,
        graph,
        noise,
        cfg,
        eval_callback=eval,
        sample_callback=sample,
        device=device,
        checkpoint_dir=checkpoint_dir
    )
    trainer.train(train_ds)


if __name__ == "__main__":
    main()