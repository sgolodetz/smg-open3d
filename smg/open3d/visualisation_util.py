import math
import numpy as np
import open3d as o3d

from itertools import product
from typing import List, Optional, Tuple

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
    def make_trajectory_segments(trajectory: List[Tuple[float, np.ndarray]], *, colour: Tuple[float, float, float]) \
            -> o3d.geometry.LineSet:
        """
        Make the line segments needed to visualise a trajectory.

        :param trajectory:  The trajectory to visualise.
        :param colour:      The colour to use for the line segments.
        :return:            The line segments.
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
        return lines

    @staticmethod
    def make_voxel_grid(mins: List[float], maxs: List[float], voxel_size: List[float]) -> o3d.geometry.LineSet:
        """
        Make a wireframe Open3D voxel grid.

        :param mins:        The minimum bounds of the voxel grid.
        :param maxs:        The maximum bounds of the voxel grid.
        :param voxel_size:  The voxel size.
        :return:            The voxel grid.
        """
        vals = {}
        for i in range(3):
            maxs[i] = mins[i] + math.ceil((maxs[i] - mins[i]) / voxel_size[i]) * voxel_size[i]
            vals[i] = np.linspace(mins[i], maxs[i], int((maxs[i] - mins[i]) / voxel_size[i]) + 1)

        xpts1 = [(mins[0], y, z) for y, z in product(vals[1], vals[2])]
        xpts2 = [(maxs[0], y, z) for y, z in product(vals[1], vals[2])]

        ypts1 = [(x, mins[1], z) for x, z in product(vals[0], vals[2])]
        ypts2 = [(x, maxs[1], z) for x, z in product(vals[0], vals[2])]

        zpts1 = [(x, y, mins[2]) for x, y in product(vals[0], vals[1])]
        zpts2 = [(x, y, maxs[2]) for x, y in product(vals[0], vals[1])]

        pts1 = xpts1 + ypts1 + zpts1
        pts2 = xpts2 + ypts2 + zpts2
        corrs = [(i, i) for i in range(len(pts1))]

        pcd1 = o3d.geometry.PointCloud()
        pcd2 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pts1)
        pcd2.points = o3d.utility.Vector3dVector(pts2)

        # noinspection PyCallByClass
        return o3d.geometry.LineSet.create_from_point_cloud_correspondences(pcd1, pcd2, corrs)

    @staticmethod
    def visualise_geometries(geoms: List[o3d.geometry.Geometry], *, axis_size: float = 0.1) -> None:
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
    def visualise_geometry(geom: o3d.geometry.Geometry, *, axis_size: float = 0.1) -> None:
        """
        Visualise an Open3D geometry.

        :param geom:        The geometry to visualise.
        :param axis_size:   The size of the coordinate axes to add.
        """
        VisualisationUtil.visualise_geometries([geom], axis_size=axis_size)

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
