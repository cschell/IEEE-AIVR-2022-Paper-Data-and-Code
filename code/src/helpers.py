from functools import lru_cache
from typing import Union

import hydra

from src.datamodules.talking_with_hands_datamodules import BinDatamodule, WindowDatamodule
import os

from src.utils import utils

log = utils.get_logger(__name__)

def load_and_setup_datamodule(datamodule_config) -> Union[BinDatamodule, WindowDatamodule]:
    log.info("setting up datamodule (no cache)")
    datamodule = hydra.utils.instantiate(datamodule_config)
    datamodule.setup()
    return datamodule


@lru_cache(maxsize=os.environ.get("DATAMODULE_LRU_CACHE_MAXSIZE", 1))
def cached_load_and_setup_datamodule(datamodule_config) -> Union[BinDatamodule, WindowDatamodule]:
    log.info("datamodule cache miss")
    return load_and_setup_datamodule(datamodule_config)
