from scipy.spatial.transform import Rotation

from src.datamodules.datasets.helpers import compute_velocities_simple, compute_change_idxs, compute_velocities_quats
import numpy as np
import pandas as pd


def test_compute_change_idxs():
    take_ids = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2])
    expected_result =   [0,       3,          7]

    change_idxs = compute_change_idxs(take_ids)

    np.testing.assert_array_equal(change_idxs, expected_result)


def test_compute_velocities_simple():
    test_take = pd.DataFrame([
        [1, 2, 3],
        [10, 20, 30],
        [100, 200, 300],
        [1000, 2000, 3000],
        [10000, 20000, 30000],
    ], columns=["test_pos_x", "test_pos_y", "test_pos_z"])

    expected_test_take_velocities_fss_1 = -np.array([
        [np.nan, np.nan, np.nan],
        [9, 18, 27],
        [90, 180, 270],
        [900, 1800, 2700],
        [9000, 18000, 27000],
    ])

    expected_test_take_velocities_fss_3 = -np.array([
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [999, 1998, 2997],
        [9990, 19980, 29970],
    ])

    first_take = pd.DataFrame(np.random.rand(11, 3))
    last_take = pd.DataFrame(np.random.rand(13, 3))

    # to make sure the method respects take boundaries we build two other takes and add
    # it before and behind the test take
    all_takes = pd.concat([first_take, test_take, last_take], ignore_index=True)
    change_idxs = np.cumsum([0] + [len(first_take), len(test_take)])

    velocities_fss_1 = compute_velocities_simple(X=all_takes, change_idxs=change_idxs, frame_step_size=1).to_numpy()
    assert np.array_equal(velocities_fss_1[change_idxs[1]:change_idxs[2]], expected_test_take_velocities_fss_1, equal_nan=True)

    velocities_fss_3 = compute_velocities_simple(X=all_takes, change_idxs=change_idxs, frame_step_size=3).to_numpy()
    assert np.array_equal(velocities_fss_3[change_idxs[1]:change_idxs[2]], expected_test_take_velocities_fss_3, equal_nan=True)


def test_compute_velocities_quats():
    # create some random test rotations and build a test dataframe from it
    test_rotations = Rotation.from_euler("xyz", np.random.randint(-180, 180, (14, 3)), degrees=True)
    test_take = pd.DataFrame(test_rotations.as_quat(), columns=["test_rot_x", "test_rot_y", "test_rot_z", "test_rot_w"])

    # to make sure the method respects take boundaries we build two other takes and add
    # it before and behind the test take
    first_take = pd.DataFrame(np.random.rand(11, 4), columns=["test_rot_x", "test_rot_y", "test_rot_z", "test_rot_w"])
    last_take = pd.DataFrame(np.random.rand(13, 4), columns=["test_rot_x", "test_rot_y", "test_rot_z", "test_rot_w"])

    all_takes = pd.concat([first_take, test_take, last_take], ignore_index=True)
    change_idxs = np.cumsum([0] + [len(first_take), len(test_take)])

    # we test frame step sizes from 1 to 4
    for frame_step_size in range(1, 4):
        # compute velocities for the current `frame_step_size`
        computed_velocities = compute_velocities_quats(X=all_takes, change_idxs=change_idxs, frame_step_size=frame_step_size).to_numpy()
        computed_test_velocities = Rotation.from_quat(computed_velocities[change_idxs[1]:change_idxs[2]][frame_step_size:])

        # when `frame_step_size` is > 1 we try different offsets to make sure every frame is tested
        for offset in range(frame_step_size):
            # when the resulting velocities in `computed_test_velocities` are consecutively applied to the
            # initial orientation, the computed final orientation should be equal to the final orientation
            # from `test_rotations`
            current_orientation = test_rotations[offset]
            for idx in range(offset, len(computed_test_velocities), frame_step_size):
                current_orientation = current_orientation * computed_test_velocities[idx]

            final_computed_orientation = current_orientation.as_euler("xyz", degrees=True).round(4)
            final_real_orientation = test_rotations[offset::frame_step_size][-1].as_euler("xyz", degrees=True).round(4)

            assert np.array_equal(final_computed_orientation, final_real_orientation)