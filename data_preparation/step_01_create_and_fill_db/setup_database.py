import psycopg2

keypoints = [
    "head",
    "left_hand",
    "right_hand",
]
keypoint_column_names = [f"{k}_pos_{c}" for k in keypoints for c in "xyz"] + \
                        [f"{k}_rot_{c}" for k in keypoints for c in "xyzw"]

def setup_database(db_kwargs):

    with psycopg2.connect(**db_kwargs) as conn:
        with conn.cursor() as curs:
            print("dropping existing data")
            curs.execute("DROP TABLE IF EXISTS frames CASCADE")
            curs.execute("DROP TABLE IF EXISTS takes CASCADE")
            print("creating tables")
            curs.execute("""CREATE TABLE takes (
                                id serial PRIMARY KEY,
                                subject_id varchar NOT NULL,
                                session_id int NOT NULL,
                                session_take_idx int NOT NULL,
                                filename varchar NOT NULL,
                                is_conversation bool NOT NULL,
                                num_frames integer
                         );""")
            curs.execute("create unique index unique_take_idx on takes using btree (subject_id, session_id, session_take_idx);")
            curs.execute("create index session_id_take_idx on takes (session_id);")
            curs.execute("create index subject_id_take_idx on takes (subject_id);")

            keypoint_column_sql = ",\n".join(["%s real NOT NULL" % (c) for c in keypoint_column_names])
            curs.execute(f"""CREATE TABLE frames (
                                frame_idx int,
                                take_id int REFERENCES takes (id),
                                {keypoint_column_sql}
                         );""")
            curs.execute("create unique index unique_frames_idx on frames (frame_idx, take_id);")
            curs.execute("create index take_id_frames_idx on frames (take_id);")
            curs.execute("create index frame_idx_frames_idx on frames (frame_idx);")
        conn.commit()