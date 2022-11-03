### Edited version of https://github.com/20tab/bvh-python

import re
from typing import Dict, List

import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd

class BvhNode:
    def __init__(self, bvh_tree, value=None, parent=None):
        self.bvh_tree = bvh_tree
        if value is None:
            value = []

        self.value = value
        self.children = []
        self.parent = parent
        if self.parent:
            self.parent.add_child(self)

    @property
    def offset(self):
        return np.array(self['OFFSET'], dtype="float32")[["xyz".index(c) for c in self.bvh_tree.translate_dims("xyz")]] * self.bvh_tree.scaling

    @property
    def is_joint(self):
        return self.value[0] == "JOINT"

    @property
    def rotation_xyz_to_column_names(self) -> Dict[str, str]:
        return {c: f"{self.name}_rot_{c}" for c in "xyzw"}

    @property
    def position_xyz_to_column_names(self) -> Dict[str, str]:
        return {c: f"{self.name}_pos_{c}" for c in "xyz"}

    @property
    def joint_children(self):
        return list(self.filter("JOINT"))

    def add_child(self, item):
        item.parent = self
        self.children.append(item)

    def filter(self, key):
        for child in self.children:
            if child.value[0] == key:
                yield child

    @property
    def is_end(self):
        return self.value[0] == 'End'

    def __iter__(self):
        for child in self.children:
            yield child

    def __getitem__(self, key):
        for child in self.children:
            for index, item in enumerate(child.value):
                if item == key:
                    if index + 1 >= len(child.value):
                        return None
                    else:
                        return child.value[index + 1:]
        raise IndexError('key {} not found'.format(key))

    def __repr__(self):
        return self.name

    @property
    def name(self):
        if len(self.value) >= 2:
            return self.value[1]
        else:
            return "n/a"


class Bvh:
    def __init__(self, data):
        self.data = data
        self.scaling = 1/10
        self.root = BvhNode(self)
        self.frames = []
        self.dims_mapping = {
            "x": "z",
            "y": "x",
            "z": "y",
        }

        self.rotation_order = "zyx" # unmapped it's yxz, as documented in https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/BVH.html

        self.tokenize()

    def tokenize(self):
        first_round = []
        accumulator = ''
        for char in self.data:
            if char not in ('\n', '\r'):
                accumulator += char
            elif accumulator:
                first_round.append(re.split('\\s+', accumulator.strip()))
                accumulator = ''
        node_stack = [self.root]
        frame_time_found = False
        node = None
        for item in first_round:
            if frame_time_found:
                self.frames.append(item)
                continue
            key = item[0]
            if key == '{':
                node_stack.append(node)
            elif key == '}':
                node_stack.pop()
            else:
                node = BvhNode(self, item)
                node_stack[-1].add_child(node)
            if item[0] == 'Frame' and item[1] == 'Time:':
                frame_time_found = True

    @property
    def real_root(self):
        return list(self.root.filter('ROOT'))[0]

    def search(self, *items):
        found_nodes = []

        def check_children(node):
            if len(node.value) >= len(items):
                failed = False
                for index, item in enumerate(items):
                    if node.value[index] != item:
                        failed = True
                        break
                if not failed:
                    found_nodes.append(node)
            for child in node:
                check_children(child)

        check_children(self.root)
        return found_nodes

    def get_joints(self):
        joints = []

        def iterate_joints(joint):
            joints.append(joint)
            for child in joint.filter('JOINT'):
                iterate_joints(child)

        iterate_joints(next(self.root.filter('ROOT')))
        return joints

    @property
    def joints(self):
        return self.get_joints()

    def get_joints_names(self):
        joints = []

        def iterate_joints(joint):
            joints.append(joint.value[1])
            for child in joint.filter('JOINT'):
                iterate_joints(child)

        iterate_joints(next(self.root.filter('ROOT')))
        return joints

    def joint_direct_children(self, name):
        joint = self.get_joint(name)
        return [child for child in joint.filter('JOINT')]

    def get_joint_index(self, name):
        return self.get_joints().index(self.get_joint(name))

    def get_joint(self, name):
        found = self.search('ROOT', name)
        if not found:
            found = self.search('JOINT', name)
        if found:
            return found[0]
        raise LookupError('joint not found')

    def joint_offset(self, name):
        joint = self.get_joint(name)
        offset = joint['OFFSET']
        return (float(offset[0]), float(offset[1]), float(offset[2]))

    def joint_channels(self, name):
        joint = self.get_joint(name)
        return joint['CHANNELS'][1:]

    def get_joint_channels_index(self, joint_name):
        index = 0
        for joint in self.get_joints():
            if joint.value[1] == joint_name:
                return index
            index += int(joint['CHANNELS'][0])
        raise LookupError('joint not found')

    def get_joint_channel_index(self, joint, channel):
        channels = self.joint_channels(joint)
        if channel in channels:
            channel_index = channels.index(channel)
        else:
            channel_index = -1
        return channel_index

    def frame_joint_channel(self, frame_index, joint, channel, value=None):
        joint_index = self.get_joint_channels_index(joint)
        channel_index = self.get_joint_channel_index(joint, channel)
        if channel_index == -1 and value is not None:
            return value
        return float(self.frames[frame_index][joint_index + channel_index])

    def frame_joint_channels(self, frame_index, joint, channels, value=None):
        values = []
        joint_index = self.get_joint_channels_index(joint)
        for channel in channels:
            channel_index = self.get_joint_channel_index(joint, channel)
            if channel_index == -1 and value is not None:
                values.append(value)
            else:
                values.append(
                        float(
                                self.frames[frame_index][joint_index + channel_index]
                        )
                )
        return values

    def frames_joint_channels(self, joint, channels, value=None):
        all_frames = []
        joint_index = self.get_joint_channels_index(joint)
        for frame in self.frames:
            values = []
            for channel in channels:
                channel_index = self.get_joint_channel_index(joint, channel)
                if channel_index == -1 and value is not None:
                    values.append(value)
                else:
                    values.append(
                            float(frame[joint_index + channel_index]))
            all_frames.append(values)
        return all_frames

    def joint_parent(self, name):
        joint = self.get_joint(name)
        if joint.parent == self.root:
            return None
        return joint.parent

    def joint_parent_index(self, name):
        joint = self.get_joint(name)
        if joint.parent == self.root:
            return -1
        return self.get_joints().index(joint.parent)

    @property
    def nframes(self):
        try:
            return int(next(self.root.filter('Frames:')).value[1])
        except StopIteration:
            raise LookupError('number of frames not found')

    @property
    def frame_time(self):
        try:
            return float(next(self.root.filter('Frame')).value[2])
        except StopIteration:
            raise LookupError('frame time not found')

    @property
    def column_names(self):
        joint_names = self.get_joints_names()

        def rename_channels(channels: List[str]):
            renamed_channels = []
            for c in channels:
                dim = str.lower(c[0])
                pos_rot = c[1:4]
                renamed_channels.append(f"{pos_rot}_{dim}")
            return renamed_channels

        columns = []
        for joint_name in joint_names:
            channels = rename_channels(self.joint_channels(joint_name))
            channels_index = self.get_joint_channels_index(joint_name)
            columns += [(channels_index + self.get_joint_channel_index(joint_name, c), f"{joint_name}_{c}") for c in
                        channels]

        column_names = [c[1] for c in sorted(columns)]

        # insert ..._rot_w column for quaternions
        for i in range(len(column_names)):
            column_name = column_names[i]
            if column_name[:len("_rot_z")] == "_rot_z":
                column_names.insert(i+1, f"{column_name[:-1]}w")

        return column_names

    @property
    def frames_df(self):
        if not hasattr(self, "_frames_df"):
            frames = np.array(self.frames, dtype="float32")
            self._frames_df = pd.DataFrame(frames, columns=self.column_names)

        return self._frames_df

    def world_coords_and_rotations(self):
        if len(self.frames_df.shape) == 1:
            num_rows = 1
            index = [self.frames_df.name]
        else:
            num_rows = self.frames_df.shape[0]
            index = self.frames_df.index

        translated_root_pos_column_names = [self.real_root.position_xyz_to_column_names[c] for c in "xyz"]
        translations = self.world_translation_for_frame(self.real_root, start_location=self.frames_df[translated_root_pos_column_names].values * self.scaling)

        world_frame = pd.DataFrame(np.zeros((num_rows, len(self.column_names))), columns=self.column_names, index=index)

        for joint, world_rotation, start_location, rotated_offset in translations:
            world_frame.loc[:, [joint.rotation_xyz_to_column_names[c] for c in "xyzw"]] = world_rotation.as_quat()
            world_frame.loc[:, [joint.position_xyz_to_column_names[c] for c in "xyz"]] = start_location
        return world_frame

    def translate_dims(self, dims):
        return [self.dims_mapping[d] for d in dims]

    def world_translation_for_frame(self, joint, world_rotation=R.identity(), start_location=np.zeros(3)):
        rotated_offset = world_rotation.apply(joint.offset)
        end_location = start_location + rotated_offset

        child_translations = []

        if not joint.is_end:
            translated_column_names = [joint.rotation_xyz_to_column_names[c] for c in self.rotation_order]
            local_rotation = R.from_euler(self.rotation_order, self.frames_df[translated_column_names].values, degrees=True)
            end_rotation = world_rotation * local_rotation

            for child_joint in joint.joint_children:
                child_translations += self.world_translation_for_frame(child_joint, world_rotation=end_rotation, start_location=end_location)

        return [(joint, world_rotation, start_location, rotated_offset)] + child_translations