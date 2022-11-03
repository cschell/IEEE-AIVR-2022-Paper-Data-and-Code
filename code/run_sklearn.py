import signal

import dotenv
import hydra
from omegaconf import DictConfig

from src.utils import utils

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

log = utils.get_logger(__name__)


@hydra.main(config_path="configs/", config_name="config_sklearn.yaml")
def main(config: DictConfig):
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from src.train_sklearn import train
    from src.utils import utils

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # Train model
    return train(config)


def trap_signals_to_handle_kills_gracefully():
    def handle_kill(signal_number, _frame):
        log.warning("getting killed")
        try:
            import wandb
            wandb.finish(exit_code=-1)
        except Exception as e:
            log.error("an error occurred during training abort:")
            log.error(e)

        signal_name = {int(s): s for s in signal.Signals}[signal_number].name
        log.warning(f"aborting because of {signal_name} signal")
        raise SystemExit(f"aborting because of {signal_name} signal")

    signal.signal(signal.SIGINT, handle_kill)
    signal.signal(signal.SIGTERM, handle_kill)


if __name__ == "__main__":
    trap_signals_to_handle_kills_gracefully()
    main()
