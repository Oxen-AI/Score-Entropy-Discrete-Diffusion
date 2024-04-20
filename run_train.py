import datetime
import os
import os.path
import gc
from itertools import chain
import yaml
from aim import Run
import oxen

import numpy as np
import torch
import torch.nn.functional as F

import data
import losses
import sampling
import graph_lib
import noise_lib
import utils
from model import SEDD
from model.ema import ExponentialMovingAverage
from transformers import GPT2LMHeadModel
import yaml
from character_tokenizer import CharacterTokenizer

def main():
    with open('configs/config.yaml', 'r') as f:
        cfg = yaml.full_load(f)

    print(cfg)

    work_dir = cfg['training']['output_dir']
    run = Run()
    run["hparams"] = cfg

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
    if device.type == "cuda":
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
    print(f"Found {os.cpu_count()} total number of CPUs.")

    # Create remote oxen repo
    repo = oxen.RemoteRepo(cfg['data']['oxen']['remote_repo'])
    if not repo.exists():
        repo.create()

    # Save config file for this run
    repo.add('configs/config.yaml')
    repo.commit("Added config file")

    # build token graph
    graph = graph_lib.get_graph(cfg, device)

    # build score model
    score_model = SEDD(cfg).to(device)

    num_parameters = sum(p.numel() for p in score_model.parameters())
    print(f"Number of parameters in the model: {num_parameters}")

    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=cfg['training']['ema'])
    print(score_model)
    print(f"EMA: {ema}")

    # build noise
    noise = noise_lib.get_noise(cfg).to(device)
    sampling_eps = 1e-5

    # build optimization state
    optimizer = losses.get_optimizer(cfg, chain(score_model.parameters(), noise.parameters()))
    print(f"Optimizer: {optimizer}")
    state = dict(optimizer=optimizer, model=score_model, noise=noise, ema=ema, step=0)

    # TODO: Strip out the scaler from the state dictionary and see how it performs

    # load in state
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, device)
    initial_step = int(state['step'])

    # load in tokenizer
    # tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    import string
    chars = string.ascii_letters # This character vocab!
    model_max_length = 2048
    tokenizer = CharacterTokenizer(chars, model_max_length)

    # Build data iterators
    train_ds, eval_ds = data.get_dataloaders(cfg)

    # print(f"Length of datasets: {len(train_ds)}, {len(eval_ds)}")

    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(cfg)
    train_step_fn = losses.get_step_fn(noise, graph, True, optimize_fn, cfg['training']['accum'])
    eval_step_fn = losses.get_step_fn(noise, graph, False, optimize_fn, cfg['training']['accum'])


    if cfg['training']['snapshot_sampling']:
        sampling_shape = (cfg['training']['batch_size'] // (cfg['ngpus'] * cfg['training']['accum']), cfg['model']['length'])
        sampling_fn = sampling.get_sampling_fn(cfg, graph, noise, sampling_shape, sampling_eps, device)

    num_train_steps = cfg['training']['n_iters']
    print(f"Starting training loop at step {initial_step}.")

    best_perplexity = float('inf')

    while state['step'] < num_train_steps + 1:
        step = state['step']


        if cfg['data']['train'] != "text8":
            batch = next(train_iter)['input_ids'].to(device)
        else:
            batch = next(train_iter).to(device)
        loss = train_step_fn(state, batch)

        run.track(loss.item(), name='loss', step=state['step'], context={ "subset":"train" })

        # flag to see if there was movement ie a full batch got computed
        if step != state['step']:
            if step % cfg['training']['log_freq'] == 0:
                print("step: %d, training_loss: %.5e" % (step, loss.item()))

            if step % cfg['training']['eval_freq'] == 0:
                if cfg['data']['valid'] != "text8":
                    eval_batch = next(eval_iter)['input_ids'].to(device)
                else:
                    eval_batch = next(train_iter).to(device)
                eval_loss = eval_step_fn(state, eval_batch)

                if step > 0:
                    print("step: %d, evaluation_loss: %.5e" % (step, eval_loss.item()))
                    run.track(eval_loss.item(), name='loss', step=state['step'], context={ "subset":"eval" })

            if step > 0 and step % cfg['training']['snapshot_freq'] == 0 or step == num_train_steps:
                # Generate and save samples
                if cfg['training']['snapshot_sampling']:
                    print(f"Generating text at step: {step}")

                    this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                    utils.makedirs(this_sample_dir)

                    ema.store(score_model.parameters())
                    ema.copy_to(score_model.parameters())
                    sample = sampling_fn(score_model)
                    ema.restore(score_model.parameters())

                    sentences = tokenizer.batch_decode(sample)

                    file_name = os.path.join(this_sample_dir, f"sample.txt")
                    with open(file_name, 'w') as file:
                        for sentence in sentences:
                            file.write(sentence + "\n")
                    repo.add(file_name)
                    repo.commit(f"Sample at step {step}")

                if cfg['eval']['perplexity']:
                    with torch.no_grad():
                        eval_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).eval()
                        batches = sample.shape[0] // cfg['eval']['perplexity_batch_size']
                        total_perplexity = 0
                        for i in range(batches):
                            s = sample[i * cfg['eval']['perplexity_batch_size']:(i + 1) * cfg['eval']['perplexity_batch_size']]
                            loss, logits = eval_model(s, labels=s)[:2]
                            logits = logits.transpose(-1, -2)
                            perplexity = F.cross_entropy(logits[..., :-1], s[..., 1:], reduction="none").mean(dim=-1).exp().mean()
                            total_perplexity += perplexity
                        total_perplexity /= batches
                        print(f"Generative Perplexity at step: {step}. Perplexity: {total_perplexity:.3f}.")

                        run.track(total_perplexity, name='perplexity', step=state['step'], context={ "subset":"eval" })

                        del eval_model, logits, loss

                        if best_perplexity < total_perplexity:
                            best_perplexity = total_perplexity
                            # write best perplexity to file
                            with open(os.path.join(work_dir, "best_perplexity.txt"), 'w') as file:
                                file.write(f"Best Perplexity: {best_perplexity:.3f} at step {step}.")
                            # save best model
                            utils.save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_best.pth'), state)
                        else:
                            # save latest model
                            utils.save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_latest.pth'), state)


if __name__ == "__main__":
    main()