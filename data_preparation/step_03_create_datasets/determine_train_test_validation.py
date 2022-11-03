import json
import pathlib

import psycopg2
import pandas as pd
from matplotlib.cm import get_cmap
import numpy as np


def determine_train_test_validation_takes(db_kwargs):
    FPS = 90

    # Selected takes need to be at least `MIN_TAKE_LENGTH` seconds long
    MIN_TAKE_LENGTH = 5 * 60 # seconds

    # Each session considered for train, validation or testing needs to have at least `MIN_NUMBER_VALID_TAKES` takes
    # that are at least `MIN_TAKE_LENGTH` seconds long
    MIN_NUMBER_VALID_TAKES = 4

    conn = psycopg2.connect(**db_kwargs)

    corrupted_take_ids_file = pathlib.Path("step_02_determine_corrupted_takes/corrupted_take_ids.json")
    if not corrupted_take_ids_file.exists():
        raise FileNotFoundError(f"could not find file with the list of corrupted take ids at {corrupted_take_ids_file};"
                                "make sure you have executed `step_02_determine_corrupted_takes/main.py` and the paths match")

    with open(corrupted_take_ids_file, "r") as file:
        corrupted_take_ids = json.load(file)

    query = "SELECT * FROM takes WHERE id NOT IN %s"
    takes = pd.read_sql(query, conn, index_col="id", params=(tuple(corrupted_take_ids), ))

    # %% determine valid sessions and valid takes

    session_id_and_session_take_idx_takes = takes.groupby(["session_id", "session_take_idx"]).size()
    takes_with_two_subjects = session_id_and_session_take_idx_takes[session_id_and_session_take_idx_takes == 2]
    takes_with_two_subjects.name = "num_subjects_in_session"
    valid_takes = takes.merge(takes_with_two_subjects, how="right", left_on=["session_id", "session_take_idx"], right_index=True)
    valid_takes = valid_takes[valid_takes.num_frames >= MIN_TAKE_LENGTH * FPS]

    valid_session_mask = (valid_takes.groupby("session_id").size() / 2) >= MIN_NUMBER_VALID_TAKES
    valid_session_ids = valid_session_mask.index[valid_session_mask].values
    valid_takes = valid_takes[valid_takes.session_id.isin(valid_session_ids)]

    # %% There are 6 subjects that appear in more than one session (called "deep" subjects by the authors of the dataset);
    #      here, we only select takes from the largest session of each deep subject

    deep_takes = valid_takes[valid_takes.subject_id.str.contains("deep")]

    deep_session_sizes = {}

    for (subject_id, session_id), take_group in deep_takes.groupby(["subject_id", "session_id"]):
        _, prev_session_size = deep_session_sizes.get(subject_id, (None, 0))

        if current_size := take_group.num_frames.sum() > prev_session_size:
            deep_session_sizes[subject_id] = (session_id, current_size)

    target_deep_sessions = {subject_id: sess[0]  for subject_id, sess in deep_session_sizes.items()}

    # %% Create visualization of the data split

    import matplotlib.pyplot as plt

    groups = valid_takes.groupby(["subject_id", "session_id"])

    fig, ax = plt.subplots(figsize=(15, 6))

    tick_labels = []
    idx = -1
    cmap = get_cmap("Paired")

    for (subject_id, session_id), group in groups:
        if len(group) < MIN_NUMBER_VALID_TAKES:
            continue

        if "deep" in subject_id and session_id != target_deep_sessions[subject_id]:
            continue
        idx +=1
        num_frames = group[group.num_frames > MIN_TAKE_LENGTH*FPS].num_frames.sort_values(ascending=False)
        for i, num_frame in enumerate(num_frames.cumsum()[::-1]):
                ax.bar([idx] , [num_frame], fc=cmap(len(num_frames) - i))

        tick_labels.append(subject_id)
    ax.set_xticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=45)

    yticks = np.arange(0, int(ax.get_ylim()[1]), 1000 * FPS)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t:.0f}" for t in (yticks/FPS)//60])
    ax.set_ylabel("length in minutes")
    plt.show()

    # %% export selected takes and data split information

    take_assignments = {}
    groups = valid_takes.groupby(["subject_id", "session_id"])
    for (subject_id, session_id), group in groups:
        if len(group) < MIN_NUMBER_VALID_TAKES:
            continue

        if "deep" in subject_id and session_id != target_deep_sessions[subject_id]:
            continue

        sorted_by_numframes = group[group.num_frames >= MIN_TAKE_LENGTH*FPS].num_frames.sort_values(ascending=True).index
        test_id, validation_id = sorted_by_numframes.tolist()[:2]
        train_ids = sorted_by_numframes[2:].tolist()

        for ids in [test_id, validation_id, train_ids]:
            assert np.all(takes.loc[ids].subject_id == subject_id)
            assert np.all(takes.loc[ids].session_id == session_id)

        take_assignments[subject_id] = {
            "session_id": int(session_id),
            "take_ids": {
                "train": train_ids,
                "validation": [validation_id],
                "test": [test_id],
            }
        }

    print(f"there are {len(take_assignments)} valid subjects")

    return take_assignments
