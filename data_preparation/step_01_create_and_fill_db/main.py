import json
import pathlib

import yaml
import re
from multiprocessing import Pool

from setup_database import setup_database
from data_aggregator import DataAggregator

SETTINGS = yaml.safe_load(open("settings.yaml"))

setup_database(SETTINGS["database"])

twh_dir = pathlib.Path(SETTINGS["talking_with_hands_directory"])

with open(twh_dir.joinpath("configs", "conversations.json"), "r") as conversations_metadata_file:
    conversations_metadata = json.load(conversations_metadata_file)

paths = list(twh_dir.glob("mocap_data/**/*.bvh"))


def process_path(args):
    idx, path = args
    session_id, session_take_idx, subject_id = re.match(r".*session(\d+)/take(\d+)/.*/.*((?:deep|shallow)\d+).*.bvh$", str(path)).groups()
    is_conversation = f"take{session_take_idx}" in conversations_metadata.get(f"session{session_id}", {})
    da = DataAggregator(SETTINGS["database"])
    da.process_bvh(path, idx, is_conversation=is_conversation)
    da.conn.close()
    del da
    return path


print("starting pool")
# don't turn processes up too high! There will be OOMs
with Pool(processes=10) as pool:
    paths = pool.map(process_path, enumerate(paths))
print("finished")
print(len(paths))
