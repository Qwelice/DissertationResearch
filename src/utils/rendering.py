from typing import Union, Optional, List

import numpy as np
import torch
import open3d as o3d
from open3d.cpu.pybind.camera import PinholeCameraParameters
from open3d.cpu.pybind.visualization import ViewControl, RenderOption
from pytorch3d.ops import cubify
from pytorch3d.renderer import look_at_view_transform, RasterizationSettings, FoVPerspectiveCameras, PointLights, \
    MeshRasterizer, SoftPhongShader, MeshRenderer, TexturesVertex
from pytorch3d.structures import Meshes

from src.structures.mesh import GenericMesh, MeshColor


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
        # noinspection PyTypeChecker
        self.visualizer.add_geometry(mesh)

        self._apply_camera_motion()

        self.visualizer.poll_events()
        self.visualizer.update_renderer()

        img = self.visualizer.capture_screen_float_buffer(do_render=True)
        img = (torch.tensor(np.asarray(img), dtype=torch.float32) * 255)
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


class VoxelRenderer:
    def __init__(self, img_size: int=256, device: Optional[Union[str, torch.device]]=None):
        self.img_size = img_size
        self.rotation = None
        self.translation = None
        self._raster_settings = RasterizationSettings(image_size=img_size, blur_radius=0.0, faces_per_pixel=1)
        self.lights_loc = [[0.0, 0.0, 3.0]]
        self._cameras = None
        self._lights = None
        self._renderer = None
        if device is None:
            device = 'cpu'
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

    def setup_cam_pos(self, distance: float, elev: float, azim: float):
        self.rotation, self.translation = look_at_view_transform(dist=distance, elev=elev, azim=azim)

    def _build_cameras_and_lights(self, num: int=1):
        if self.rotation is None or self.translation is None:
            self.rotation, self.translation = look_at_view_transform(dist=3.0, elev=30, azim=60)
        self.rotation = self.rotation.expand((num, -1, -1))
        self.translation = self.translation.expand((num, -1))
        self.lights_loc = [self.lights_loc[0] for _ in range(num)]
        self._cameras = FoVPerspectiveCameras(R=self.rotation, T=self.translation, device=self.device)
        self._lights = PointLights(location=self.lights_loc, device=self.device)

    def _build_renderer(self, num: int=1):
        if self._cameras is None or self._lights is None:
            self._build_cameras_and_lights(num)
        rasterizer = MeshRasterizer(cameras=self._cameras, raster_settings=self._raster_settings)
        shader = SoftPhongShader(device=self.device, cameras=self._cameras, lights=self._lights)
        self._renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

    def render_voxel(self,
                     voxel: torch.Tensor,
                     color: Optional[MeshColor]=None):
        voxel = (voxel > 0.5).to(torch.float32)
        if voxel.ndim == 5:
            voxel = voxel.squeeze(1)
        if color is None:
            color = MeshColor.GREEN
        mesh = cubify(voxel, thresh=0.1, device=self.device)
        verts = []
        faces = []
        verts_features = []
        for m in mesh:
            m = GenericMesh.create_from_mesh(m, device=self.device)
            if m.verts.size(0) == 0:
                continue
            m.to_center()
            m.normalize_()
            m.change_color(color)
            m = m.as_pytorch3d()
            verts.append(m.verts_packed())
            faces.append(m.faces_packed())
            verts_features.append(m.textures.verts_features_packed())
        if len(verts) == 0:
            return None
        mesh = Meshes(verts=verts, faces=faces, textures=TexturesVertex(verts_features=verts_features))
        num_mesh = len(mesh)
        if self._renderer is None:
            self._build_renderer(num_mesh)
        img = self._renderer(mesh)
        img = img[:,...,:3]
        return img

    def render_voxels(self, voxels: Union[List[torch.Tensor], tuple[torch.Tensor]], color: Optional[MeshColor]=None):
        images = []
        for vox in voxels:
            img = self.render_voxel(vox, color=color)
            if img is not None:
                images.append(img)
        return images