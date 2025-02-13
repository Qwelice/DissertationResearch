from typing import Union, Optional, List

import numpy as np
import torch
import open3d as o3d
from open3d.cpu.pybind.camera import PinholeCameraParameters
from open3d.cpu.pybind.visualization import ViewControl, RenderOption

from src.structures.mesh import GenericMesh


class Renderizer:
    def __init__(self, img_size: int):
        self.img_size = img_size
        self.visualizer = o3d.visualization.Visualizer()
        self.visualizer.create_window(visible=False, width=self.img_size, height=self.img_size)
        self.opt: RenderOption = self.visualizer.get_render_option()
        self.opt.background_color = np.array([0.87, 0.85, 0.88])
        self.opt.light_on = True
        self.ctr: ViewControl = self.visualizer.get_view_control()
        self._angles: Optional[Union[tuple, List]] = None
        self._translation: Optional[Union[tuple, List]] = None

    def _render_mesh(self, mesh: GenericMesh) -> torch.Tensor:
        self.visualizer.clear_geometries()
        mesh = mesh.as_open3d()
        self.visualizer.add_geometry(mesh)

        self._apply_camera_motion()

        self.visualizer.poll_events()
        self.visualizer.update_renderer()

        img = self.visualizer.capture_screen_float_buffer(do_render=True)
        img = (torch.tensor(np.asarray(img), dtype=torch.float32) * 255).to(dtype=torch.uint8)
        return img

    def _apply_camera_motion(self):
        parameters: PinholeCameraParameters = self.ctr.convert_to_pinhole_camera_parameters()
        new_extrinsic = parameters.extrinsic.copy()
        if self._angles is not None:
            rotation = o3d.geometry.get_rotation_matrix_from_xyz(self._angles)
            new_extrinsic[:3, :3] = new_extrinsic[:3, :3] @ rotation
            if self._translation is not None:
                translation = np.array(self._translation, dtype=np.float32)
                new_extrinsic[:3, 3] = new_extrinsic[:3, 3] + rotation @ translation
        elif self._translation is not None:
            translation = np.array(self._translation, dtype=np.float32)
            new_extrinsic[:3, 3] = new_extrinsic[:3, 3] + translation

        parameters.extrinsic = new_extrinsic
        self.ctr.convert_from_pinhole_camera_parameters(parameters)

    def setup_camera_motion(self,
                            angles: Optional[Union[tuple, List]]=None,
                            translation: Optional[Union[tuple, List]]=None,
                            radians: bool=False):
        if angles is not None:
            if len(angles) != 3:
                raise ValueError('length of angles must be 3')
            if not radians:
                angles = torch.tensor(angles, dtype=torch.float32)
                angles = torch.deg2rad(angles)
                angles = angles.cpu().tolist()
        self._angles = angles
        self._translation = translation

    def __call__(self, mesh: GenericMesh) -> torch.Tensor:
        return self._render_mesh(mesh)

    def __del__(self):
        self.visualizer.destroy_window()