
import torch
from tqdm import tqdm

from ..trainer.loss import step_fn

class Evaluator:
    def __init__(self, dataset, run, cfg, device = 'cuda'):
        self.dataset = dataset
        self.run = run
        self.cfg = cfg
        self.device = device
        
    def evaluate(self, state):
        step = state['step']
        sum_loss = 0
        print(f"Evaluating model on validation set")
        for batch in tqdm(self.dataset):
            batch = batch.to(self.device)
            loss = self.evaluate_batch(state, batch)
            sum_loss += loss.item()
        avg_loss = sum_loss / len(self.dataset)
        print("step: %d, evaluation_loss: %.5e" % (step, avg_loss))
        self.run.track(avg_loss, name='loss', step=state['step'], context={ "subset":"eval" })
        return avg_loss
    
    def evaluate_batch(self, state, batch):
        model = state['model']
        model.eval()
        with torch.no_grad():
            eval_loss = step_fn(self.cfg, state, batch, train=False)
            return eval_loss