import numpy as np
import open3d as o3d

from typing import Optional, Tuple


class VisualisationUtil:
    """TODO"""

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
    def visualise_geometry(geom: o3d.geometry.Geometry) -> None:
        """
        TODO

        :param geom:    TODO
        """
        # Set up the visualisation.
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        render_option: o3d.visualization.RenderOption = vis.get_render_option()
        render_option.line_width = 10

        # noinspection PyTypeChecker
        vis.add_geometry(geom)
        VisualisationUtil.add_axis(vis, np.eye(4), size=0.1)

        # Set the initial pose for the visualiser.
        params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        m = np.eye(4)
        params.extrinsic = m
        vis.get_view_control().convert_from_pinhole_camera_parameters(params)

        # Run the visualiser.
        vis.run()
