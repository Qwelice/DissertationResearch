from typing import Union, Optional, List

import torch
from pytorch3d.ops import cubify
from pytorch3d.renderer import look_at_view_transform, RasterizationSettings, FoVPerspectiveCameras, PointLights, \
    MeshRasterizer, SoftPhongShader, MeshRenderer, TexturesVertex
from pytorch3d.structures import Meshes

from research.utils.structures.generic_mesh import GenericMesh, MeshColor

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
        voxel = (voxel > 0.5).to(dtype=torch.float32, device=self.device)
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