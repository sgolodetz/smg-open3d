import copy
import numpy as np
import open3d as o3d

from typing import List, Optional, Tuple

from open3d.cpu.pybind.geometry import Geometry, TriangleMesh
from smg.utility import GeometryUtil


class VisualisationUtil:
    """Utility functions related to Open3D visualisations."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def add_axis(vis: o3d.visualization.Visualizer, pose: np.ndarray, *,
                 colour: Optional[Tuple[float, float, float]] = None, size: float = 1.0) -> None:
        """
        Add to the specified Open3D visualisation a set of axes for the specified pose.

        :param vis:     The Open3D visualisation.
        :param pose:    The pose (specified in camera space).
        :param colour:  An optional colour with which to paint the axes.
        :param size:    The size to give the axes (defaults to 1).
        """
        # noinspection PyArgumentList
        axes: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        if colour is not None:
            axes.paint_uniform_color(colour)
        axes.transform(pose)
        # noinspection PyTypeChecker
        vis.add_geometry(axes)

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

    # @staticmethod
    # def make_geometries_for_trajectory(trajectory: np.ndarray, colour: Tuple[float, float, float]) -> List[Geometry]:
    #     """
    #     Make the line segments needed to visualise a trajectory.
    #
    #     :param trajectory:  The trajectory to visualise.
    #     :param colour:      The colour to use for the line segments.
    #     :return:            A list containing the line segments.
    #     """
    #     frame_count = len(trajectory)
    #     line_indices = np.array(list(zip(np.arange(frame_count - 1), np.arange(1, frame_count))))
    #     colours = [colour for _ in range(len(line_indices))]
    #     lines = o3d.geometry.LineSet(
    #         points=o3d.utility.Vector3dVector(trajectory[:, :, -1]),
    #         lines=o3d.utility.Vector2iVector(line_indices),
    #     )
    #     lines.colors = o3d.utility.Vector3dVector(colours)
    #     return [lines]

    @staticmethod
    def make_geometries_for_trajectory(trajectory: List[Tuple[float, np.ndarray]],
                                       colour: Tuple[float, float, float]) -> List[Geometry]:
        """
        Make the line segments needed to visualise a trajectory.

        :param trajectory:  The trajectory to visualise.
        :param colour:      The colour to use for the line segments.
        :return:            A list containing the line segments.
        """
        length: int = len(trajectory)
        points: List[np.ndarray] = [pose[0:3, 3] for _, pose in trajectory]
        line_indices: np.ndarray = np.array(list(zip(np.arange(length - 1), np.arange(1, length))))
        colours: List[Tuple[float, float, float]] = [colour for _ in range(len(line_indices))]
        lines: o3d.geometry.LineSet = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(line_indices),
        )
        lines.colors = o3d.utility.Vector3dVector(colours)
        return [lines]

    @staticmethod
    def visualise_geometries(geoms: List[Geometry], *, axis_size: float = 0.1) -> None:
        """
        Visualise some Open3D geometries.

        :param geoms:       The geometries to visualise.
        :param axis_size:   The size of the coordinate axes to add.
        """
        # Set up the visualisation.
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        render_option: o3d.visualization.RenderOption = vis.get_render_option()
        render_option.line_width = 10

        for geom in geoms:
            # noinspection PyTypeChecker
            vis.add_geometry(geom)

        VisualisationUtil.add_axis(vis, np.eye(4), size=axis_size)

        # Set the initial pose for the visualiser.
        params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        m = np.eye(4)
        params.extrinsic = m
        vis.get_view_control().convert_from_pinhole_camera_parameters(params)

        # Run the visualiser.
        vis.run()

    @staticmethod
    def visualise_geometry(geom: o3d.geometry.Geometry) -> None:
        """
        Visualise an Open3D geometry.

        :param geom:    The geometry to visualise.
        """
        VisualisationUtil.visualise_geometries([geom])

    @staticmethod
    def visualise_rgbd_image(colour_image: np.ndarray, depth_image: np.ndarray,
                             intrinsics: Tuple[float, float, float, float]) -> None:
        """
        Visualise an RGB-D image in 3D.

        :param colour_image:    The colour image.
        :param depth_image:     The depth image.
        :param intrinsics:      The camera intrinsics.
        """
        # Make a coloured point cloud from the RGB-D image.
        depth_mask: np.ndarray = np.where(depth_image != 0, 255, 0).astype(np.uint8)
        pcd_points, pcd_colours = GeometryUtil.make_point_cloud(colour_image, depth_image, depth_mask, intrinsics)

        # Convert it to Open3D format.
        pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_points)
        pcd.colors = o3d.utility.Vector3dVector(pcd_colours)

        # Visualise it.
        VisualisationUtil.visualise_geometry(pcd)
