from typing import List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import LightningLoggerBase

from src.callbacks.log_best_metrics_callback import LogBestMetrics
from src.callbacks.user_halt_callback import UserHaltCallback
from src.helpers import cached_load_and_setup_datamodule, load_and_setup_datamodule
from src.utils import utils
from src.wandb.helpers import upload_config_and_checkpoints_to_wandb, is_wandb_logger_enabled

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Init lightning loggers
    loggers = initialize_loggers(config)

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        log.info(f"seeding everything to {config.seed}")
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    if config.get("cache_datamodule", False):
        log.info(f"Instantiating datamodule (caching) <{config.datamodule._target_}>")
        # this caches the datamodule, which saves some time during stage 1 of the hp search
        datamodule = cached_load_and_setup_datamodule(config.datamodule)
    else:
        log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
        datamodule = load_and_setup_datamodule(config.datamodule)

    log.info("saving data stats")
    datamodule.safe_train_data_stats_settings()

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")

    model = hydra.utils.instantiate(config.model, num_features=datamodule.train_dataset.num_features,
                                    num_out_classes=datamodule.train_dataset.num_classes)

    identifier: LightningModule = hydra.utils.instantiate(config.lightning_module,
                                                          loss_weights=datamodule.train_dataset.loss_weights,
                                                          model=model,
                                                          metrics=config.monitored_metrics.metrics,
                                                          labels=datamodule.train_dataset.labels)

    # Init lightning callbacks
    callbacks = initialize_callbacks(config, datamodule)
    callbacks += [UserHaltCallback()]
    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")

    trainer: Trainer = hydra.utils.instantiate(
            config.trainer,
            callbacks=callbacks,
            logger=loggers,
    )

    # Figure out batch size so we can utilize the GPU as well as possible
    datamodule.batch_size = config.datamodule.batch_size
    log.info(f"batch size was {datamodule.batch_size}")
    log.info("starting batch size tuning")
    trainer.tune(model=identifier, datamodule=datamodule, scale_batch_size_kwargs={"init_val": datamodule.batch_size, "mode": "power", "max_trials": 5})

    # limit batch size to 100,000 samples, otherwise we run into memory problems
    datamodule.batch_size = min(datamodule.batch_size, 100_000)
    config.datamodule.batch_size = min(datamodule.batch_size, 100_000)  # this is only for logging purposes

    log.info("batch size tuning finished")
    log.info(f"batch size is now {datamodule.train_dataloader().batch_size}")

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
            config=config,
            model=identifier,
            trainer=trainer,
    )

    try:
        # Train the model
        log.info("starting training")
        trainer.fit(model=identifier, datamodule=datamodule)
        log.info("training finished")
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            # this signals Optuna that the run failed, but does
            # not abort the job (at least that's what the docs say)
            log.error(str(e))
            log.warning("aborting run")
            return float('nan')
        else:
            raise e

    if is_wandb_logger_enabled(loggers):
        upload_config_and_checkpoints_to_wandb(datamodule)

    finalize_train(loggers)

    # Return metric score for hyperparameter optimization with optuna
    if optimize_metric := config.get("optimize_metric"):
        optimize_direction = config["optimize_direction"]
        min_max = optimize_direction[:3]
        best_metric = identifier.best_logged_metrics[(optimize_metric, min_max)]
        log.info(f"telling optuna best value for {optimize_metric} was {best_metric}")
        return best_metric


def initialize_callbacks(config, datamodule):
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))
    if "monitored_callbacks" in config:
        callbacks += build_checkpoint_callback_list(config.monitored_metrics.metrics, store_path="early_stopping")

    callbacks += [LogBestMetrics()]
    return callbacks


def build_checkpoint_callback_list(checkpoint_metrics, store_path):
    checkpoint_metric_callbacks = []
    for metric in checkpoint_metrics:
        metric_id = "{name}_{dataset}".format(**metric)

        checkpoint_metric_callbacks.append(ModelCheckpoint(monitor=metric_id,
                                                           save_top_k=1,
                                                           mode=metric['mode'],
                                                           dirpath=store_path,
                                                           filename=f"{metric['mode']}_{metric_id}"
                                                           ))
    return checkpoint_metric_callbacks


def initialize_loggers(config):
    loggers: List[LightningLoggerBase] = []
    if "logger" in config:
        for logger_name, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating loggers <{lg_conf._target_}>")
                logger = hydra.utils.instantiate(lg_conf)
                loggers.append(logger)
    return loggers


def finalize_train(loggers):
    # tell wandb to finish to avoid problems during hp search
    if is_wandb_logger_enabled(loggers):
        import wandb
        wandb.finish()
