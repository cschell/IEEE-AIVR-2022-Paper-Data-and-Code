import numpy as np
import pandas as pd

from src.data_selection import DataSelection
from src.datamodules.datasets.bin_maker import BinMaker


def test_single_frame_package_maker():
    frames = pd.DataFrame([
        [1, 4, 40],
        [1, 5, 50],
        [1, 6, 60],
        [1, 7, 70],
        [1, 8, 80],
    ])

    expected_packages = [
        # mean              std                                             median              min                 max
        [1.0, 4.5, 45.0, 0.0, 0.7071067811865476, 7.0710678118654755, 1.0, 4.5, 45.0, 1.0, 4.0, 40.0, 1.0, 5.0, 50.0],
        [1.0, 5.5, 55.0, 0.0, 0.7071067811865476, 7.0710678118654755, 1.0, 5.5, 55.0, 1.0, 5.0, 50.0, 1.0, 6.0, 60.0],
        [1.0, 6.5, 65.0, 0.0, 0.7071067811865476, 7.0710678118654755, 1.0, 6.5, 65.0, 1.0, 6.0, 60.0, 1.0, 7.0, 70.0],
        [1.0, 7.5, 75.0, 0.0, 0.7071067811865476, 7.0710678118654755, 1.0, 7.5, 75.0, 1.0, 7.0, 70.0, 1.0, 8.0, 80.0]
    ]

    bin_maker = BinMaker(frames=frames, take_ids=np.array([0] * len(frames)), data_selection=DataSelection.SCENE_RELATIVE)
    result_packages, _frame_ids = bin_maker.make(frames_per_package=2)
    np.testing.assert_array_equal(expected_packages, result_packages)


def test_single_frame_package_maker_multi_takes():
    frames = pd.DataFrame([
        [1, 4, 40],
        [1, 5, 50],
        [1, 6, 60],
        [1, 7, 70],
        [1, 8, 80],

        [5, -4, -40],
        [5, -6, -60],
        [5, -8, -80],
    ])

    take_ids = np.array([0] * 5 + [1] * 3)

    expected_packages = [
        # mean          std                                          median          min             max
        [1, 4.5, 45, 0, 0.7071067811865476, 7.0710678118654755, 1, 4.5, 45, 1, 4, 40, 1, 5, 50],
        [1, 5.5, 55, 0, 0.7071067811865476, 7.0710678118654755, 1, 5.5, 55, 1, 5, 50, 1, 6, 60],
        [1, 6.5, 65, 0, 0.7071067811865476, 7.0710678118654755, 1, 6.5, 65, 1, 6, 60, 1, 7, 70],
        [1, 7.5, 75, 0, 0.7071067811865476, 7.0710678118654755, 1, 7.5, 75, 1, 7, 70, 1, 8, 80],

        [5, -5, -50, 0, 1.4142135623730951, 14.142135623730951, 5, -5, -50, 5, -6, -60, 5, -4, -40],
        [5, -7, -70, 0, 1.4142135623730951, 14.142135623730951, 5, -7, -70, 5, -8, -80, 5, -6, -60]
    ]

    expected_frame_ids = [1, 2, 3, 4, 6, 7]

    bin_maker = BinMaker(frames=frames, take_ids=take_ids, data_selection=DataSelection.SCENE_RELATIVE)
    result_packages, result_frame_ids = bin_maker.make(frames_per_package=2)

    np.testing.assert_array_equal(expected_packages, result_packages)
    np.testing.assert_array_equal(expected_frame_ids, result_frame_ids)
