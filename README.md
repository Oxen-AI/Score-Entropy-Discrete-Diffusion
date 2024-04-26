# Text Diffusion Models with SEDD

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repo is based on the [Score Entropy Discrete Diffusion](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) repository. It contains a PyTorch implementation for the paper [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution
](https://arxiv.org/abs/2310.16834) by [Aaron Lou](https://aaronlou.com), [Chenlin Meng](https://cs.stanford.edu/~chenlin/) and [Stefano Ermon](https://cs.stanford.edu/~ermon/).

![cover](assets/language_diffusion_forward.gif)


It has been modified to perform character level diffusion for a few practical reasons.

1) Be able to run on a single A10 GPU
2) See how well a character level language diffusion model performs.


## Installation

Simply run

```
python -m venv ~/.venv_sedd
source ~/.venv_sedd/bin/activate
pip install -r requirements.txt
```

This will create a virtual environmet ```~/.venv_sedd``` environment with packages installed. 

Note that this installs with CUDA 11.8, and different CUDA versions must be installed manually. The biggest factor is making sure that the ```torch``` and ```flash-attn``` packages use the same CUDA version (more found [here](https://github.com/Dao-AILab/flash-attention)).

## Working with Pretrained Models

### Download Models

I uploaded the raw PyTorch `SEDD-large` model to Oxen.ai for convenience. To download you can simply run:

```
oxen clone https://hub.oxen.ai/models/SEDD-large
```

This repository contains both the model weights `checkpoint.pth` and the `config.yaml` file with other necessary parameters.


### Run Sampling

We can run sampling using a command 

```
python scripts/run_sample.py --model /path/to/SEDD-large --steps 128
```

We can also sample conditionally using

```
python scripts/run_sample_cond.py --model_path MODEL_PATH --step STEPS --prefix PREFIX --suffix SUFFIX
```

## Training New Models

### Run Training

We provide training code, which can be run with the command

```
python scripts/run_train.py --repo oxen_username/repo_name
```

## Acknowledgements

This repository builds heavily off of [Score Entropy Discrete Diffusion](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion), [score sde](https://github.com/yang-song/score_sde_pytorch), [plaid](https://github.com/igul222/plaid), and [DiT](https://github.com/facebookresearch/DiT).

It was stripped down for demo and learning purposes for [arXiv dives](https://www.oxen.ai/community) series by Oxen.ai.