import numpy as np
import open3d as o3d


# MAIN CLASS

class ReconstructionUtil:
    """Utility functions related to 3D reconstruction."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def integrate_frame(colour_image: np.ndarray, depth_image: np.ndarray, world_to_camera: np.ndarray,
                        intrinsics: o3d.camera.PinholeCameraIntrinsic,
                        tsdf: o3d.pipelines.integration.ScalableTSDFVolume,
                        *, depth_trunc: float = 4.0) -> None:
        """
        Integrate the specified frame into the TSDF.

        :param colour_image:        The frame's colour image.
        :param depth_image:         The frame's depth image.
        :param world_to_camera:     The frame's pose (as a transformation from world space to camera space).
        :param intrinsics:          The camera intrinsics.
        :param tsdf:                The TSDF.
        :param depth_trunc:         The depth truncation value (depths greater than this are ignored).
        """
        # Check that the colour and depth images are the same size.
        if colour_image.shape[:2] != depth_image.shape:
            raise RuntimeError("Cannot integrate the frame into the TSDF: the images are different sizes")

        # Prepare the RGB-D image that will be integrated into the TSDF.
        # noinspection PyArgumentList, PyCallByClass
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(colour_image),
            o3d.geometry.Image(depth_image.astype(np.float32)),
            depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False
        )

        # Integrate the RGB-D image into the TSDF.
        tsdf.integrate(rgbd_image, intrinsics, world_to_camera)

    @staticmethod
    def make_mesh(tsdf: o3d.pipelines.integration.ScalableTSDFVolume, *, print_progress: bool = False) \
            -> o3d.geometry.TriangleMesh:
        """
        Make a triangle mesh from the specified TSDF.

        :param tsdf:            The TSDF.
        :param print_progress:  Whether or not to print out progress messages.
        :return:                The triangle mesh.
        """
        if print_progress:
            print("Extracting a triangle mesh from the TSDF")

        mesh: o3d.geometry.TriangleMesh = tsdf.extract_triangle_mesh()

        if print_progress:
            print("Computing vertex normals for the mesh")

        mesh.compute_vertex_normals()

        return mesh
