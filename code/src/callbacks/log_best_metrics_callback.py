from pytorch_lightning import Trainer, Callback

from src.lightning_module.identifier import Identifier
from src.wandb.helpers import is_wandb_logger_enabled


class LogBestMetrics(Callback):
    def on_validation_end(self, trainer: Trainer, identifier: Identifier):
        self._add_best_logged_metrics_to_wandb(trainer, identifier)

    def _add_best_logged_metrics_to_wandb(self, trainer: Trainer, identifier: Identifier):
        if is_wandb_logger_enabled(trainer.logger):
            import wandb

            for (metric_name, direction), best_value in identifier.best_logged_metrics.items():
                target_direction = "min" if "loss" in metric_name else "max"

                if target_direction == direction:
                    wandb.run.summary[f"best_{metric_name}"] = best_value
