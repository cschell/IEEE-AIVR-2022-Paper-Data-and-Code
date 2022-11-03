import os
import pathlib
from typing import Optional

import hydra
import wandb
from omegaconf import DictConfig
from pytorch_lightning import (
    seed_everything,
)
from pytorch_lightning.loggers import WandbLogger

from src.helpers import cached_load_and_setup_datamodule, load_and_setup_datamodule
from src.metrics import DatasetPurpose, get_metric_name
from src.sklearn_random_forest_trainer import SKLearnRandomForestTrainer
from src.utils import utils

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # currently for the random forest training only WandB is supported
    logger: WandbLogger = hydra.utils.instantiate(config.logger.wandb)

    logger.log_hyperparams({
        "datamodule": config["datamodule"],
        "model": config["model"],
        "code_dir": config["work_dir"],
        "work_dir": pathlib.Path.cwd(),
        "seed": config.get("seed", None),
        "node_name": os.environ.get("NODE_NAME", None),
    })

    # Init lightning datamodule
    if config.get("cache_datamodule", False):
        log.info(f"Instantiating datamodule (caching) <{config.datamodule._target_}>")
        # this caches the datamodule, which saves some time during stage 1 of the hp search
        datamodule = cached_load_and_setup_datamodule(config.datamodule)
    else:
        log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
        datamodule = load_and_setup_datamodule(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")

    model = hydra.utils.instantiate(config.model)

    checkpoint_dir = pathlib.Path().cwd().joinpath("checkpoints")
    checkpoint_dir.mkdir(parents=True)

    trainer = SKLearnRandomForestTrainer(datamodule.train_dataset,
                                         datamodule.val_dataset,
                                         save_dir_path=checkpoint_dir,
                                         model=model)

    log.info("start training")
    trainer.fit()
    log.info("training complete")

    log.info("computing scores")
    train_metrics = trainer.compute_train_scores()
    validation_metrics = trainer.compute_validation_scores()

    for metric_name, value in train_metrics.items():
        wandb.log({
            f"best_{get_metric_name(metric_name, DatasetPurpose.TRAIN)}": value
        })

    for metric_name, value in validation_metrics.items():
        wandb.log({
            f"best_{get_metric_name(metric_name, DatasetPurpose.VALIDATION)}": value
        })

    if checkpoint_dir.exists():
        wandb.save(f"{checkpoint_dir}/*.pkl*")
    else:
        log.warning(f"checkpoint directory {checkpoint_dir} does not exists, so no checkpoints will be uploaded to wandb.io")

    if datamodule.data_stats_path.exists():
        wandb.save(str(datamodule.data_stats_path))
    else:
        log.warning(f"data stats file with training stats {datamodule.data_stats_path} does not exists, so it cannot be uploaded to wandb.io")
    wandb.save(".hydra/*.yaml")

    wandb.finish()

    # Return metric score for hyperparameter optimization with optuna
    if optimize_metric := config.get("optimize_metric"):
        best_metric = [v for m, v in validation_metrics.items() if get_metric_name(m, DatasetPurpose.VALIDATION) == optimize_metric][0]
        log.info(f"telling optuna best value for {optimize_metric} was {best_metric}")
        return best_metric
