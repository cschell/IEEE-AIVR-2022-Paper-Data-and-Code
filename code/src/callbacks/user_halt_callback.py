import signal

from pytorch_lightning import Callback

from src.utils import utils

log = utils.get_logger(__name__)


class UserHaltCallback(Callback):
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_start(trainer, pl_module)

        def handle_kill(signal_number, _frame):
            signal_name = {int(s): s for s in signal.Signals}[signal_number].name
            log.warning(f"received {signal_name} signal, halting training without raising an error")
            trainer.should_stop = True

        signal.signal(signal.SIGUSR1, handle_kill)
