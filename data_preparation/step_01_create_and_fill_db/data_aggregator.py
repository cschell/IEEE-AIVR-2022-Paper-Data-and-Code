import re
from datetime import datetime
import pandas as pd
import psycopg2

import bvh
import numpy as np

from setup_database import keypoint_column_names


class DataAggregator:
    required_features = {
        "b_head": "head",
        "b_l_wrist_twist": "left_hand",
        "b_r_wrist_twist": "right_hand",
    }

    def __init__(self, db_kwargs):
        self.required_features_inv = {v: k for k, v in self.required_features.items()}
        self.conn = psycopg2.connect(**db_kwargs)

    def _log(self, txt):
        print(txt)

    def process_bvh(self, bvh_path, idx=0, is_conversation=False):
        self._log(f"{idx: 03d} | start working on {bvh_path} | {datetime.now().isoformat()}")

        session_id, session_take_idx, subject_id = re.match(r".*session(\d+)/take(\d+)/.*/.*((?:deep|shallow)\d+).*.bvh$", str(bvh_path)).groups()

        self._log(f"{idx: 03d} | reading bvh | {datetime.now().isoformat()}")
        with open(bvh_path) as f:
            mocap = bvh.Bvh(f.read())

        frames = mocap.world_coords_and_rotations()

        self._log(f"{idx: 03d} | start db operations | {datetime.now().isoformat()}")
        with self.conn.cursor() as curs:
            curs.execute("INSERT INTO takes (filename, subject_id, session_id, session_take_idx, is_conversation) VALUES (%s, %s, %s, %s, %s) RETURNING id;",
                         (str(bvh_path), subject_id, int(session_id), int(session_take_idx), is_conversation))

            take_id = curs.fetchone()[0]

            values_sql_list = []

            for frame_idx, row in frames.iterrows():
                values_sql_list.append(curs.mogrify("(%s, %%s, %%s)" % ",".join(["%s"] * len(keypoint_column_names)),
                                                    [frame_idx, take_id] + [row[self.required_features_inv[kcn[:-len("_pos_x")]] + kcn[-len("_pos_x"):]] for kcn in
                                                                            keypoint_column_names]).decode('utf-8'))

            curs.execute(f"""INSERT INTO frames (frame_idx, take_id, {",".join(keypoint_column_names)}) VALUES {",".join(values_sql_list)}""")

            update_video_frame_count_query = "UPDATE takes set num_frames = %s WHERE id = %s"
            curs.execute(update_video_frame_count_query, (len(frames), take_id))
            self._log(f"{idx: 03d} | committing | {datetime.now().isoformat()}")
            self.conn.commit()

        self.conn.commit()
        assert np.any(pd.read_sql("SELECT * FROM takes WHERE session_id = %s AND session_take_idx = %s", params=(session_id, session_take_idx), con=self.conn))

        self._log(f"{idx: 03d} | finished working on {bvh_path} | {datetime.now().isoformat()}")

    def __del__(self):
        try:
            self.conn.close()
        except:
            pass