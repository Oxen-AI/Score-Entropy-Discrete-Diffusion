
import abc
import torch
from tqdm import tqdm

from sedd.models.catsample import sample_categorical

def score_fn(model, x, sigma):
    sigma = sigma.reshape(-1)
    score = model(x, sigma)
    # when sampling return true score (not log used for training)
    return score.exp()

class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, model, x, t, step_size):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass

class EulerPredictor(Predictor):
    def update_fn(self, model, x, t, step_size):
        sigma, dsigma = self.noise(t)
        score = score_fn(model, x, sigma)

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(x, rev_rate)
        return x

class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, model, x, t):
        sigma = self.noise(t)[0]

        score = score_fn(model, x, sigma)
        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        # truncate probabilities
        if self.graph.absorb:
            probs = probs[..., :-1]
        
        #return probs.argmax(dim=-1)
        return sample_categorical(probs)

@torch.no_grad()
def pc_sampler(cfg, model, graph, noise, steps, batch_size=1, projector = lambda x: x, device='cuda', eps=1e-5, denoise=True):
    predictor = EulerPredictor(graph, noise)
    denoiser = Denoiser(graph, noise)

    batch_dims = (batch_size, cfg['model']['length'])

    x = graph.sample_limit(*batch_dims).to(device)
    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    dt = (1 - eps) / steps

    print(f"Sampling with {steps} steps")
    for i in tqdm(range(steps)):
        t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
        x = projector(x)
        x = predictor.update_fn(model, x, t, dt)

    if denoise:
        # denoising step
        x = projector(x)
        t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
        x = denoiser.update_fn(model, x, t)
        
    return x

class Sampler:
    def __init__(self, cfg, device='cuda'):
        self.cfg = cfg
        self.device = device

    def sample(self, tokenizer, model, graph, noise, batch_size=1, steps=1024):
        cfg = self.cfg
        sample = pc_sampler(cfg, model, graph, noise, steps, batch_size, device=self.device)
        sentences = tokenizer.batch_decode(sample)
        return sentences