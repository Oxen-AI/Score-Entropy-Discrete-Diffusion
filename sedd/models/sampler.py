
import os
import oxen

from sampling import get_sampling_fn

class Sampler:
    def __init__(self, tokenizer, sample_dir, cfg, device='cuda'):
        self.tokenizer = tokenizer
        self.sample_dir = sample_dir
        self.cfg = cfg
        self.device = device
        
    def sample(self, state):
        step = state['step']
        model = state['model']
        graph = state['graph']
        noise = state['noise']
        print(f"Generating text at step: {step}")
        
        sampling_eps = 1e-5
        cfg = self.cfg

        batch_size = cfg['eval']['batch_size']
        context_length = cfg['model']['length']
        sampling_shape = (batch_size, context_length)
        sampling_fn = get_sampling_fn(cfg, graph, noise, sampling_shape, sampling_eps, self.device)

        this_sample_dir = os.path.join(self.sample_dir, "iter_{}".format(step))
        os.makedirs(this_sample_dir, exist_ok=True)

        sample = sampling_fn(model)
        sentences = self.tokenizer.batch_decode(sample)

        file_name = os.path.join(this_sample_dir, f"sample.txt")
        with open(file_name, 'w') as file:
            for sentence in sentences:
                file.write(sentence + "\n")
                file.write("="*80 + "\n")
        
        repo = oxen.RemoteRepo(cfg['data']['remote_repo'])
        repo.add(file_name)
        repo.commit(f"Sample at step {step}")