import numpy as np

from argparse import ArgumentParser
from typing import List, Tuple

from smg.open3d.visualisation_util import VisualisationUtil
from smg.utility.trajectory_util import TrajectoryUtil


def main():
    relocaliser_trajectory: List[Tuple[float, np.ndarray]] = TrajectoryUtil.load_tum_trajectory(
        "C:/smglib/smg-relocalisation/examples/relocaliser_trajectory.txt"
    )

    tracker_trajectory: List[Tuple[float, np.ndarray]] = TrajectoryUtil.load_tum_trajectory(
        "C:/smglib/smg-relocalisation/examples/tracker_trajectory.txt"
    )

    relocaliser_geoms = VisualisationUtil.make_geometries_for_trajectory(relocaliser_trajectory, (0.0, 1.0, 0.0))
    tracker_geoms = VisualisationUtil.make_geometries_for_trajectory(tracker_trajectory, (1.0, 0.0, 0.0))
    VisualisationUtil.visualise_geometries(relocaliser_geoms + tracker_geoms, axis_size=0.1)


if __name__ == "__main__":
    main()


# import numpy as np
# import open3d as o3d
# import os
#
# from open3d.cpu.pybind.geometry import Geometry
# from typing import List
#
# from smg.open3d.visualisation_util import VisualisationUtil
# from smg.utility.trajectory_util import TrajectoryUtil
#
#
# def main():
#     # # Parse any command-line arguments and construct a settings object.
#     # parser = ArgumentParser()
#     # Settings.add_args_to_parser(parser)
#     # args = parser.parse_args()
#     # settings: Settings = Settings(args)
#     #
#     # # Load the trajectories and the frame timestamps.
#     # orb_slam_dir: str = os.path.join(settings.get("root_dir"), settings.get("sequence_name"), "orb_slam")
#
#     orb_slam_dir: str = "D:/datasets/kitti_raw/2011_09_26/2011_09_26_drive_0005_sync/orb_slam"
#
#     camera_traj_filename: str = os.path.join(orb_slam_dir, "CameraTrajectory.txt")
#     camera_traj: np.ndarray = TrajectoryUtil.load_trajectory(camera_traj_filename)
#     camera_timestamps: List[float] = TrajectoryUtil.load_tum_timestamps(camera_traj_filename)
#
#     gt_traj_filename: str = os.path.join(orb_slam_dir, "GTCameraTrajectory.txt")
#     gt_traj: np.ndarray = TrajectoryUtil.load_trajectory(gt_traj_filename)
#     gt_timestamps: List[float] = TrajectoryUtil.load_tum_timestamps(gt_traj_filename)
#
#     keyframe_timestamps: List[float] = TrajectoryUtil.load_tum_timestamps(
#         os.path.join(orb_slam_dir, "KeyFrameTrajectory.txt")
#     )
#
#     # Make the Open3D geometries needed.
#     camera_geoms: List[Geometry] = VisualisationUtil.make_geometries_for_trajectory(camera_traj, colour=(0, 0, 1))
#     gt_geoms: List[Geometry] = VisualisationUtil.make_geometries_for_trajectory(gt_traj, colour=(0, 0.75, 0))
#     keyframe_geoms: List[Geometry] = VisualisationUtil.make_geometries_for_keyframes(
#         keyframe_timestamps, camera_timestamps, gt_timestamps, camera_traj, gt_traj
#     )
#
#     # Set up the visualisation.
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name="ORB-SLAM Trajectory Visualiser", width=800, height=600)
#
#     # Add coordinate axes at the origin to anchor the visualisation.
#     # noinspection PyArgumentList
#     axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
#     vis.add_geometry(axes)
#
#     # Add the geometries to the visualisation.
#     for geom in camera_geoms + gt_geoms + keyframe_geoms:
#         # noinspection PyTypeChecker
#         vis.add_geometry(geom)
#
#     # Set the initial pose for the visualiser.
#     params = vis.get_view_control().convert_to_pinhole_camera_parameters()
#     m = np.eye(4)
#     params.extrinsic = m
#     vis.get_view_control().convert_from_pinhole_camera_parameters(params)
#
#     # Run the visualiser.
#     vis.run()
#
#
# if __name__ == "__main__":
#     main()
