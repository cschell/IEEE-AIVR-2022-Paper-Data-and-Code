import logging
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.spatial.transform import Rotation

from src.metrics import get_metric_name

logger = logging.getLogger("fail safe")

from src.utils import utils

log = utils.get_logger(__name__)


def fail_safe(fun):
    def inner(*args, **kwargs):
        try:
            fun(*args, **kwargs)
        except Exception as e:
            logger.warning("there is an error in a fail safe method:")
            logger.exception(e)
            logger.warning("ignoring this error.")

    return inner


def compute_change_idxs(values, add_last_idx=False):
    if values.dtype == np.dtype("object"):
        values = pd.Categorical(values).codes

    change_idxs = np.where(np.diff(values, prepend=np.nan))[0]

    if add_last_idx:
        change_idxs = np.concatenate([change_idxs, [len(values)]])

    return change_idxs


def limit_data_per_subject(data_df, max_number_of_frames_per_subject):
    change_idxs_for_training_data_limiting = compute_change_idxs(
        data_df.subject_id, add_last_idx=True
    )
    log.warning(f"WARNING: LIMITING TRAINING DATA")
    log.warning(f"data size before filtering: {len(data_df)}")
    frame_filter = np.zeros((len(data_df)), dtype="bool")
    for change_idx_idx in range(1, len(change_idxs_for_training_data_limiting)):
        a = change_idxs_for_training_data_limiting[change_idx_idx - 1]
        b = change_idxs_for_training_data_limiting[change_idx_idx]

        limit_idx = min(a + max_number_of_frames_per_subject, b)
        frame_filter[a:limit_idx] = True
    data_df = data_df[frame_filter]
    log.warning(f"data size after filtering: {len(data_df)}")
    return data_df


def build_checkpoint_callback_list(checkpoint_metrics, dir_path):
    checkpoint_metric_callbacks = []
    for metric, purpose, mode in checkpoint_metrics:
        metric_name = get_metric_name(metric, purpose)

        checkpoint_metric_callbacks.append(
            ModelCheckpoint(
                monitor=metric_name,
                save_top_k=(10 if metric_name == "winners_val" else 1),
                mode=mode,
                dirpath=dir_path,
                filename=f"{mode}_{metric_name}",
            )
        )
    return checkpoint_metric_callbacks


def compute_relative_positions_and_rotations(
    frames,
    target_joints,
    reference_joint="head",
    *,
    coordinate_system: Optional[Dict[str, str]] = None,
):
    if not coordinate_system:
        FORWARD = 0
        RIGHT = 1
        UP = 2
    else:
        FORWARD = "xyz".index(coordinate_system["forward"])
        RIGHT = "xyz".index(coordinate_system["right"])
        UP = "xyz".index(coordinate_system["up"])

    assert FORWARD != RIGHT != UP

    # assert FORWARD == 0 and RIGHT == 1 and UP == 2

    FORWARD_DIRECTION = np.identity(3)[FORWARD]
    UP_DIRECTION = np.identity(3)[UP]

    reference_columns = sorted(
        [
            c
            for c in frames.columns
            if c.startswith(reference_joint + "_") and "_pos_" in c
        ]
    )
    target_columns = sorted(
        [c for c in frames.columns if c not in reference_columns and "_pos_" in c]
    )
    reference_pos_columns = [f"{reference_joint}_pos_{xyz}" for xyz in "xyz"]

    num_samples = len(frames)

    ## parse rotations of the reference joint (the head)
    reference_rotation_names = [f"{reference_joint}_rot_{c}" for c in "xyzw"]
    reference_rotations = Rotation.from_quat(frames[reference_rotation_names])

    ## retrieve projection of viewing direction of the reference joint on
    ## the horizontal plane by first applying the head rotation onto the
    ## forward vector and then zeroing out the UP axis
    horizontal_plane_projections = reference_rotations.apply(FORWARD_DIRECTION)
    horizontal_plane_projections[:, UP] = 0

    ## compute angle between projection and forward vectors using dot product and arccos
    rotations_around_up_axis = np.arccos(
            (horizontal_plane_projections @ FORWARD_DIRECTION) /
            (np.linalg.norm(FORWARD_DIRECTION) * np.linalg.norm(horizontal_plane_projections, axis=1))
    )

    ## compute correction rotation
    # find out into which direction the vectors have to be rotated
    correction_rotation_directions = -np.sign(horizontal_plane_projections[:, RIGHT])

    # build euler angle rotation vector for rotation around UP axis
    # (usage of `.from_euler` feels a bit hacky, but that's easier than building
    # a rotation matrix from scratch)
    correction_rotations_raw = np.zeros((num_samples, 3))
    correction_rotations_raw[:, UP] = correction_rotation_directions * rotations_around_up_axis
    correction_rotations = Rotation.from_euler("xyz", correction_rotations_raw, degrees=False)

    ## apply correction positions and rotations
    relative_positions_and_rotations = pd.DataFrame()

    for joint_name in target_joints: # => joint_name is either `right_hand` or `left_hand`
        # apply rotations to position vector of `joint_name`
        joint_position_names = [f"{joint_name}_pos_{c}" for c in "xyz"]

        shifted_positions = frames[joint_position_names].values - frames[reference_pos_columns].values
        shifted_and_rotated_positions = correction_rotations.apply(shifted_positions)
        relative_positions_and_rotations[joint_position_names] = shifted_and_rotated_positions

        # rotate the world rotation of `joint_name` by the correction rotation and save quaternion representations
        joint_rotation_names = [f"{joint_name}_rot_{c}" for c in "xyzw"]
        rotated_rotations = correction_rotations * Rotation.from_quat(frames[joint_rotation_names])
        relative_positions_and_rotations[joint_rotation_names] = rotated_rotations.as_quat()

    # add horizontal rotations of reference joint
    relative_positions_and_rotations[reference_rotation_names] = (correction_rotations * reference_rotations).as_quat()

    return relative_positions_and_rotations


def calc_invalid_frames(frame_step_size, change_idxs):
    return np.concatenate([np.array(change_idxs) + step for step in range(frame_step_size)])


def compute_velocities_simple(X: pd.DataFrame, frame_step_size: int, change_idxs: Iterable[int]) -> pd.DataFrame:
    velocities = X.copy()

    velocities[frame_step_size:] = velocities.values[:-frame_step_size] - velocities.values[frame_step_size:]
    velocities[:frame_step_size] = np.nan

    invalid_frames = calc_invalid_frames(frame_step_size, change_idxs)

    velocities.values[invalid_frames, :] = np.nan
    velocities = velocities.add_prefix("delta_")
    return velocities


def compute_velocities_quats(X: pd.DataFrame, frame_step_size: int, change_idxs: Iterable[int]) -> pd.DataFrame:
    velocities = X.copy()

    rotation_columns = [c for c in X.columns if "_rot_" in c]
    assert np.all(rotation_columns == X.columns)

    velocities[:] = np.nan

    joint_names = set([c[:-len("_rot_x")] for c in rotation_columns])

    for joint_name in joint_names:
        joint_rotation_names = [f"{joint_name}_rot_{c}" for c in "xyzw"]
        try:
            rot = Rotation.from_quat(X[joint_rotation_names])
        except:
            print(X[joint_rotation_names])
            raise
        delta_rot = rot[:-frame_step_size].inv() * rot[frame_step_size:]
        velocities.loc[frame_step_size:, joint_rotation_names] = delta_rot.as_quat()

    assert not np.all(np.isnan(velocities[rotation_columns][frame_step_size:]))

    invalid_frames = calc_invalid_frames(frame_step_size, change_idxs)

    velocities.values[invalid_frames, :] = np.nan
    velocities = velocities.add_prefix("delta_")
    return velocities


def compute_velocities_for_position_and_rotations(X, frame_step_size, change_idxs) -> pd.DataFrame:
    velocities = X.copy()

    position_columns = [c for c in X.columns if "_pos_" in c]
    rotation_columns = [c for c in X.columns if "_rot_" in c]

    velocities[position_columns] = compute_velocities_simple(X[position_columns], frame_step_size, change_idxs)
    velocities[rotation_columns] = compute_velocities_quats(X[rotation_columns], frame_step_size, change_idxs)

    return velocities
