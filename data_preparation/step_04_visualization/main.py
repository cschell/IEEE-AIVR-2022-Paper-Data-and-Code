import matplotlib
import psycopg2
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from step_03_create_datasets.determine_train_test_validation import determine_train_test_validation_takes

cm = 1 / 2.54
LATEX_LINEWIDTH_INCHES = 8.83159 * cm

SETTINGS = yaml.safe_load(open("settings.yaml"))

take_assignments = determine_train_test_validation_takes(SETTINGS["database"])


matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "figure.titlesize": 8,
    "legend.title_fontsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 300
})

take_ids = [take_id for subject_takes in take_assignments.values() for take_ids in subject_takes["take_ids"].values() for take_id in take_ids]


query = "SELECT * FROM takes WHERE id IN %s"
with psycopg2.connect(**SETTINGS["database"]) as conn:
    takes = pd.read_sql(query, conn, index_col="id", params=(tuple(take_ids), ))

uses = pd.Series(dtype="str", index=takes.index)

for subject_name, info in take_assignments.items():
    for use, ids in info["take_ids"].items():
        for le_id in ids:
            uses.loc[le_id] = use

takes["use"] = pd.Categorical(uses, categories=["train", "validation", "test"])

data = takes.groupby(["subject_id", "use"], as_index=False).sum()
used_takes = data

subjects = used_takes.subject_id.str.extract(r"([^\d]+)(\d+)").astype({1: int})

used_takes["formatted subject_ids"] = subjects[0].str.cat(subjects[1].apply(lambda x: f" {x:02d}"))
used_takes = used_takes.sort_values("formatted subject_ids")



fig, ax = plt.subplots(figsize=(LATEX_LINEWIDTH_INCHES, 5))

cmap = ListedColormap(["#003f5c", "#bc5090", "#ffa600"])

padding = 2

scatter = ax.scatter(y=used_takes.subject_id.factorize()[0] + (used_takes.use.factorize()[0] / 2 - .5) / padding,
           x=used_takes.num_frames,
           c=used_takes.use.factorize()[0],
           cmap=cmap,
           zorder=3,
           s=10)
ax.barh(y=used_takes.subject_id.factorize()[0]+ (used_takes.use.factorize()[0] / 2 - .5) / padding,
            width="num_frames",
            data=used_takes,
            color=cmap(used_takes.use.factorize()[0]),
            align="center",
            alpha=0.2,
            zorder=2,
            height=0.1)

ax.yaxis.set_tick_params(length=0)
ax.set_ylim(-1, 34)
ax.set_yticks(range(0, 34))
ax.set_yticklabels(range(1, 35))

ax.set_ylabel("")

FPS = 90
xticks = np.arange(10 * FPS * 60, int(ax.get_xlim()[1]), FPS * 60 * 10)
ax.set_xticks(xticks)
ax.xaxis.set_tick_params(length=0)
ax.set_xticklabels([f"{t:.0f}" for t in (xticks/FPS)//60])
ax.set_xlabel("length in minutes")
ax.xaxis.grid(True, zorder=1, linestyle=":")
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.suptitle("Overview Dataset: Per Subject Data Split")
patches = []

for idx in range(34):
    height = 0.8
    width = 0.5 * 90 * 60
    offset = - width / 3
    r = Rectangle((offset, idx - height/2), width=width, height=height, clip_on=False)
    patches.append(r)

pc = PatchCollection(patches, facecolor="#bbb", alpha=1, linewidth=0, clip_on=False, zorder=4)

ax.add_collection(pc)

legend1 = ax.legend(scatter.legend_elements()[0], ["train", "validation", "test"],
                    loc="upper right", fancybox=False, frameon=False)
ax.add_artist(legend1)


fig.tight_layout()
fig.savefig("../uaixdlamds/tmp/report/dataset_split.pgf")
fig.savefig("../uaixdlamds/tmp/report/dataset_split.pgf")
plt.show()
