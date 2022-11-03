import pathlib
import psycopg2
import pandas as pd
from tqdm import tqdm
import os


def create_train_validation_datasets(take_assignments, output_path, db_kwargs):
    conn = psycopg2.connect(**db_kwargs)

    target_keypoint_columns = [
        "head_pos_x",
        "head_pos_y",
        "head_pos_z",
        "head_rot_x",
        "head_rot_y",
        "head_rot_z",
        "head_rot_w",
        "left_hand_pos_x",
        "left_hand_pos_y",
        "left_hand_pos_z",
        "left_hand_rot_x",
        "left_hand_rot_y",
        "left_hand_rot_z",
        "left_hand_rot_w",
        "right_hand_pos_x",
        "right_hand_pos_y",
        "right_hand_pos_z",
        "right_hand_rot_x",
        "right_hand_rot_y",
        "right_hand_rot_z",
        "right_hand_rot_w",
    ]

    output_path = pathlib.Path(output_path)
    if output_path.exists():
        os.remove(output_path)

    for subject_id, dataset_info in tqdm(take_assignments.items(), leave=False):
        session_id = dataset_info["session_id"]

        take_ids_by_dataset = dataset_info["take_ids"]

        for dataset, take_ids in take_ids_by_dataset.items():
            for take_id in take_ids:
                query = f"SELECT frame_idx, take_id, {','.join(target_keypoint_columns)} FROM frames WHERE take_id = {take_id} ORDER BY frame_idx"
                frames = pd.read_sql(query, conn)
                frames["subject_id"] = subject_id
                frames["session_id"] = session_id

                frames.dropna().to_hdf(output_path, key=dataset, mode="a", index=False, dropna=True, append=True, min_itemsize=9)

