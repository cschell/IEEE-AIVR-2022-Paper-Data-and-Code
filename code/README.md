# Machine Learning Code

## Description

This repository contains code for the models and training routine.

We use the configuration framework [Hydra](https://hydra.cc) along with [Hydra Lightning Template](https://github.com/ashleve/lightning-hydra-template). You will therefore find all relevant configurations in [configs/](configs/).

## Requirements

- this codebase has been tested on an Ubuntu 20.04 machine and with the included [Dockerfile](Dockerfile)
- you need at least Python 3.7
- a current CUDA version that works with the used PyTorch version

## Data

You find the dataset used for training in `data/talking_with_hands_data.hdf5`. It has been generated with the code in `../data_preparation`.

## Prerequesites

Install python packages with

```bash
pip install -r requirements.txt
```

## Run training

### Single training

We provide the configurations to train each of the final architectures in [configs/experiment/](configs/experiment/). To train a model with one of these configurations, use this
command:

```bash
python run.py experiment=<config name>

python run.py experiment=mlp_brv

python run.py experiment=rnn_br

# we use scikit-learn for the random forest implementation, so use `run_sklearn.py` here
python run_sklearn.py experiment=random_forest
```

### Hyperparameter Search

We rely on [Optuna](https://optuna.org/) for the hyperparameter search. This project uses the [optuna-sweeper plugin](https://hydra.cc/docs/next/plugins/optuna_sweeper/), which
links hydra and optuna. You find the configurations we used for stage 1 and 2 in *configs/hparams_search/* and can perform the hyperparameter for one model+data selection search
like this:

```bash
python run.py -m hparams_search=gru_brv

# or for random forest
python run_sklearn.py -m hparams_search=rf_brv
```

Optuna is configured to use a SQLite database, so you can run multiple jobs at once and they coordinate through the database. If you would like to configure another RDBMS, have a
look at [configs/hparams_search/_defaults.yaml](configs/hparams_search/_defaults.yaml).

### Notes

- we use [Weights & Biases](https://wandb.ai) to monitor training runs; you can configure a different logger in configs/config.yaml, but note that this has noted been tested.

## Support

If you have questions or need help, feel free to contact [Christian Schell](christian.schell@uni-wuerzburg.de).

