import copy
import numpy as np
import open3d as o3d

from open3d.cpu.pybind.geometry import Geometry, TriangleMesh
from scipy.spatial.transform import Rotation
from typing import List, Tuple

from smg.utility.geometry_util import GeometryUtil


class TrajectoryUtil:
    """Utility functions related to trajectories."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def load_trajectory(filename: str, *, canonicalise_poses: bool = False) -> np.ndarray:
        """
        Load a KITTI or TUM trajectory from a file.

        :param filename:            The name of the file containing the trajectory.
        :param canonicalise_poses:  Whether or not to canonicalise the poses in the trajectory (i.e. transform each
                                    pose by the inverse of the first pose in the trajectory to ensure that the new
                                    first pose is the identity).
        :return:                    The trajectory (as an n*3*4 numpy array, where n is the sequence length).
        """
        # Read in all of the lines in the file.
        with open(filename, "r") as f:
            lines = f.read().split("\n")

        # Convert the lines into an n*m float array, where n is the sequence length and m is the number of elements
        # in each line. The file will either be in KITTI format, in which case each line will contain a 3*4 rigid body
        # transform (in row major order), and m will be 12, or it will be in TUM format, in which case each line will
        # contain a timestamp, translation vector and quaternion, and be of the form "timestamp tx ty tz qx qy qz qw",
        # and m will be 8.
        transforms: np.ndarray = np.array([list(map(float, line.split(" "))) for line in lines if line])

        # noinspection PyUnusedLocal
        new_transforms: np.ndarray

        if transforms.shape[1] == 12:
            # If the file was in KITTI format, the n*12 array can simply be reshaped to n*3*4.
            new_transforms = transforms.reshape((-1, 3, 4))
        else:
            # If the file was in TUM format, we need to determine the 3*4 matrix manually for each frame.
            new_transforms = np.zeros((transforms.shape[0], 3, 4))
            for frame_idx, transform in enumerate(transforms):
                new_transforms[frame_idx] = np.eye(4)[0:3, :]
                new_transforms[frame_idx, 0:3, 3] = transform[1:4]
                r: Rotation = Rotation.from_quat(transform[4:])
                new_transforms[frame_idx, 0:3, 0:3] = r.as_matrix()

        if canonicalise_poses:
            TrajectoryUtil.transform_trajectory(
                new_transforms, pre=np.linalg.inv(GeometryUtil.to_4x4(new_transforms[0]))
            )

        return new_transforms

    @staticmethod
    def load_tum_timestamps(filename: str) -> List[float]:
        """
        Load the frame timestamps for a TUM trajectory from a file.

        :param filename:    The name of the file containing the trajectory.
        :return:            The frame timestamps for the trajectory.
        """
        # Read in all of the lines in the file.
        with open(filename, "r") as f:
            lines = f.read().split("\n")

        # Make a list of the timestamps and return them.
        timestamps: List[float] = []
        for line in lines:
            if line:
                timestamp: float = float(line.split(" ")[0])
                timestamps.append(timestamp)

        return timestamps

    @staticmethod
    def make_geometries_for_keyframes(keyframe_timestamps: List[float], camera_timestamps: List[float],
                                      gt_timestamps: List[float], camera_traj: np.ndarray, gt_traj: np.ndarray) \
            -> List[Geometry]:
        """
        Make the connected axes needed to visualise the keyframes for the camera and ground truth trajectories.

        :param keyframe_timestamps: The timestamps for the keyframe trajectory.
        :param camera_timestamps:   The timestamps for the camera trajectory.
        :param gt_timestamps:       The timestamps for the ground truth trajectory.
        :param camera_traj:         The camera trajectory.
        :param gt_traj:             The ground truth trajectory.
        :return:                    A list containing the connected axes.
        """
        geoms = []

        # noinspection PyArgumentList
        base_axis: TriangleMesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

        camera_idx: int = 0
        gt_idx: int = 0

        # For each keyframe:
        for keyframe_idx, keyframe_timestamp in enumerate(keyframe_timestamps):
            # Find the relevant camera and ground truth poses.
            while camera_idx < len(camera_timestamps) and camera_timestamps[camera_idx] < keyframe_timestamp:
                camera_idx += 1
            while gt_idx < len(gt_timestamps) and gt_timestamps[gt_idx] < keyframe_timestamp:
                gt_idx += 1
            if camera_idx == len(camera_timestamps) or gt_idx == len(gt_timestamps):
                break
            poses: List[np.ndarray] = [camera_traj[camera_idx], gt_traj[gt_idx]]

            # Add one axis for the camera keyframe and one for the ground truth keyframe.
            for pose in poses:
                r, t = pose[0:3, 0:3], pose[0:3, 3]
                axis: TriangleMesh = copy.deepcopy(base_axis)
                # noinspection PyArgumentList
                axis.rotate(r, center=(0, 0, 0))
                axis.translate(t)
                geoms.append(axis)

            # Add a line segment connecting the two axes, to show the error.
            line = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(np.vstack(list(pose[0:3, 3] for pose in poses))),
                lines=o3d.utility.Vector2iVector(np.array([0, 1], ndmin=2)),
            )
            line.paint_uniform_color([1, 0, 0])
            geoms.append(line)

        return geoms

    @staticmethod
    def make_geometries_for_trajectory(trajectory: np.ndarray, colour: Tuple[float, float, float]) -> List[Geometry]:
        """
        Make the line segments needed to visualise a trajectory.

        :param trajectory:  The trajectory to visualise.
        :param colour:      The colour to use for the line segments.
        :return:            A list containing the line segments.
        """
        frame_count = len(trajectory)
        line_indices = np.array(list(zip(np.arange(frame_count - 1), np.arange(1, frame_count))))
        colours = [colour for _ in range(len(line_indices))]
        lines = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(trajectory[:, :, -1]),
            lines=o3d.utility.Vector2iVector(line_indices),
        )
        lines.colors = o3d.utility.Vector3dVector(colours)
        return [lines]

    @staticmethod
    def transform_trajectory(trajectory: np.ndarray, *, pre: np.ndarray = np.eye(4), post: np.ndarray = np.eye(4)) \
            -> None:
        """
        Apply the specified rigid-body transforms to each pose in a trajectory (in-place).

        :param trajectory:  The trajectory whose poses should be transformed.
        :param pre:         The rigid-body transform with which to pre-multiply each pose (expressed as a 4*4 matrix).
        :param post:        The rigid-body transform with which to post-multiply each pose (expressed as a 4*4 matrix).
        """
        for frame_idx in range(trajectory.shape[0]):
            trajectory[frame_idx] = GeometryUtil.to_3x4(pre @ GeometryUtil.to_4x4(trajectory[frame_idx]) @ post)

    @staticmethod
    def write_tum_pose(f, timestamp: float, pose: np.ndarray) -> None:
        """
        Write a timestamped pose to a TUM trajectory file.

        :param f:           The TUM trajectory file.
        :param timestamp:   The timestamp.
        :param pose:        The pose.
        """
        r: Rotation = Rotation.from_matrix(pose[0:3, 0:3])
        t: np.ndarray = pose[0:3, 3]
        f.write(" ".join([str(timestamp)] + list(map(str, t)) + list(map(str, r.as_quat()))))
        f.write("\n")
