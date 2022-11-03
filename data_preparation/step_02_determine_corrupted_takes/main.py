import json
import pathlib

import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle

SETTINGS = yaml.safe_load(open("settings.yaml"))

# This query computes a score for each take that provides clues about the take being corrupted or not;
# this score is composed of the standard deviation of both hands on the z (up) axis – corrupted takes only show
# a small jitter for each keypoint, so this becomes visible in the score
query = """SELECT takes.id, session_id, session_take_idx, subject_id, num_frames / 90 / 60, score
            FROM (SELECT take_id, STDDEV(right_hand_pos_z) + STDDEV(left_hand_pos_z) as score FROM frames GROUP BY take_id) as stats, takes
            WHERE stats.take_id = takes.id"""

with psycopg2.connect(**SETTINGS["database"]) as conn:
    takes = pd.read_sql(query, conn, index_col="id").sort_values("score")

takes.to_csv("takes_with_corruption_scores.csv", index=False)

# %% takes with a score lower than 0.5 seem to be corrupted

CORRUPTED_TAKES_THRESHOLD = 0.5
corrupted_takes = takes[takes.score < CORRUPTED_TAKES_THRESHOLD]

# %% Create visualization of the score distribution

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].scatter(x=np.arange(len(takes)), y=takes.score)
axes[0].set_ylabel("score")
axes[0].set_xlabel("number of takes")
axes[0].grid()
axes[0].add_patch(Rectangle((-20,-0.4), width=280, height=2.3, edgecolor="red", lw=4, fill=None))
axes[1].scatter(x=np.arange(len(takes)), y=takes.score)
axes[1].set_ylim((0, 2))
axes[1].set_xlim((0, 250))
axes[1].set_xlabel("number of takes")
axes[0].set_title("scores")
axes[1].set_title("scores (zoomed)")
axes[1].grid()

fig.tight_layout()
plt.show()

# %% writing corrupt take ids to disk, so that we can exclude them later

output_path = pathlib.Path(__file__).parent.joinpath("corrupted_take_ids.json")

print(f"writing IDs of corrupted takes to {output_path}")
with open(output_path, "w") as file:
    json.dump(sorted(corrupted_takes.index.tolist()), file, indent=2)