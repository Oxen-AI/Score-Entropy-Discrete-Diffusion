
from .loss import step_fn
import torch.optim as optim
from itertools import chain
import os
import torch

class Trainer:
    def __init__(self, run, model, graph, noise, config, eval_callback=None, sample_callback=None, device='cuda', checkpoint_dir='checkpoints'):
        self.graph = graph
        self.model = model
        self.noise = noise
        self.config = config
        self.eval_callback = eval_callback
        self.sample_callback = sample_callback
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.run = run

    def train(self, dataset):
        cfg = self.config

        # build optimization state
        optimizer = optimizer = optim.AdamW(
            chain(self.model.parameters(), self.noise.parameters()),
            lr=cfg['optim']['lr'],
            betas=(cfg['optim']['beta1'],
                cfg['optim']['beta2']),
            eps=cfg['optim']['eps'],
            weight_decay=cfg['optim']['weight_decay']
        )

        state = dict(
            optimizer=optimizer,
            model=self.model,
            noise=self.noise,
            graph=self.graph,
            step=0
        )

        n_epochs = cfg['training']['n_epochs']
        for e in range(n_epochs):
            print(f"Epoch {e}")
            for batch in dataset:
                self.step(state, batch)

    def step(self, state, batch):
        cfg = self.config
        step = state['step']

        batch = batch.to(self.device)
        loss = step_fn(cfg, state, batch, train=True)

        self.run.track(loss.item(), name='loss', step=state['step'], context={ "subset":"train" })

        # flag to see if there was movement ie a full batch got computed
        if step % cfg['training']['log_freq'] == 0:
            print("step: %d, training_loss: %.5e" % (step, loss.item()))

        if step % cfg['training']['eval_freq'] == 0:
            if self.eval_callback is not None:
                self.eval_callback(state)

        if step % cfg['training']['snapshot_freq'] == 0:
            torch.save(state['model'].state_dict(), os.path.join(self.checkpoint_dir, f'checkpoint.pth'))

        if step > 0 and step % cfg['training']['snapshot_freq'] == 0:
            # Generate and save samples
            if self.sample_callback is not None:
                self.sample_callback(state)
