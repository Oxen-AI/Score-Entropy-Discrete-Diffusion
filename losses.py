import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import graph_lib
from model import utils as mutils


def get_loss_fn(noise, graph, train, sampling_eps=1e-3, lv=False):

    def loss_fn(model, batch, cond=None, t=None, perturbed_batch=None):
        """
        Batch shape: [B, L] int. D given from graph
        """

        if t is None:
            if lv:
                raise NotImplementedError("Yeah I gotta do this later")
            else:
                t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps
            
        sigma, dsigma = noise(t)
        
        if perturbed_batch is None:
            perturbed_batch = graph.sample_transition(batch, sigma[:, None])

        log_score_fn = mutils.get_score_fn(model, train=train, sampling=False)
        log_score = log_score_fn(perturbed_batch, sigma)
        loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)

        loss = (dsigma[:, None] * loss).sum(dim=-1)

        return loss

    return loss_fn


def get_optimizer(config, params):
    if config['optim']['optimizer'] == 'Adam':
        optimizer = optim.Adam(params, lr=config['optim']['lr'], betas=(config['optim']['beta1'], config['optim']['beta2']), eps=config['optim']['eps'],
                               weight_decay=config['optim']['weight_decay'])
    elif config['optim']['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(params, lr=config['optim']['lr'], betas=(config['optim']['beta1'], config['optim']['beta2']), eps=config['optim']['eps'],
                               weight_decay=config['optim']['weight_decay'])
    else:
        raise NotImplementedError(
            f"Optimizer {config['optim']['optimizer']} not supported yet!")

    return optimizer

def get_step_fn(noise, graph, train, config):
    loss_fn = get_loss_fn(noise, graph, train)

    accum_iter = 0
    total_loss = 0

    def step_fn(state, batch, cond=None):
        nonlocal accum_iter 
        nonlocal total_loss

        model = state['model']
        warmup = config['optim']['warmup']
        accum = config['training']['accum']
        lr = config['optim']['lr']
        step = state['step']
        grad_clip = config['optim']['grad_clip']

        if train:
            optimizer = state['optimizer']
            loss = loss_fn(model, batch, cond=cond).mean() / accum
            
            loss.backward()

            accum_iter += 1
            total_loss += loss.detach()
            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                if warmup > 0:
                    for g in optimizer.param_groups:
                        g['lr'] = lr * np.minimum(step / warmup, 1.0)
                if grad_clip >= 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

                optimizer.step()
                
                optimizer.zero_grad()
                
                loss = total_loss
                total_loss = 0
        else:
            with torch.no_grad():
                loss = loss_fn(model, batch, cond=cond).mean()

        return loss

    return step_fn