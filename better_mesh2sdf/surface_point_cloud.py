import trimesh
import logging
logging.getLogger("trimesh").setLevel(9000)
import numpy as np
from sklearn.neighbors import KDTree
import math
from mesh_to_sdf.utils import sample_uniform_points_in_unit_sphere
from mesh_to_sdf.utils import get_raster_points, check_voxels

class BadMeshException(Exception):
    pass


class SurfacePointCloud:
    def __init__(self, mesh, points, normals=None):
        self.mesh = mesh
        self.points = points
        self.normals = normals

        self.kd_tree = KDTree(points)

    def get_random_surface_points(self, count):
        return self.mesh.sample(count)

    def get_sdf(self, query_points):
        distances, indices = self.kd_tree.query(query_points, k=1)
        distances = distances.astype(np.float32)

        return distances

    def get_sdf_in_batches(self, query_points, batch_size=1000000):
        if query_points.shape[0] <= batch_size:
            return self.get_sdf(query_points)

        n_batches = int(math.ceil(query_points.shape[0] / batch_size))
        batches = [
            self.get_sdf(points)
            for points in np.array_split(query_points, n_batches)
        ]
        return np.concatenate(batches) # distances

    def get_voxels(self, voxel_resolution, pad=False, check_result=False):
        result = self.get_sdf_in_batches(get_raster_points(voxel_resolution))

        sdf = result
        voxels = sdf.reshape((voxel_resolution, voxel_resolution, voxel_resolution))

        if check_result and not check_voxels(voxels):
            raise BadMeshException()

        if pad:
            voxels = np.pad(voxels, 1, mode='constant', constant_values=1)

        return voxels

    def sample_sdf_near_surface(self, number_of_points=500000):
        query_points = []
        surface_sample_count = number_of_points
        surface_points = self.get_random_surface_points(surface_sample_count)
        query_points.append(surface_points + np.random.normal(scale=0.0025, size=(surface_sample_count, 3)))
        query_points.append(surface_points + np.random.normal(scale=0.00025, size=(surface_sample_count, 3)))

        query_points = np.concatenate(query_points).astype(np.float32)

        sdf = self.get_sdf_in_batches(query_points)
        return query_points, sdf


def get_equidistant_camera_angles(count):
    increment = math.pi * (3 - math.sqrt(5))
    for i in range(count):
        theta = math.asin(-1 + 2 * i / (count - 1))
        phi = ((i + 1) * increment) % (2 * math.pi)
        yield phi, theta


def sample_from_mesh(mesh, sample_point_count=10000000, calculate_normals=True):
    if calculate_normals:
        points, face_indices = mesh.sample(sample_point_count, return_index=True)
        normals = mesh.face_normals[face_indices]
    else:
        points = mesh.sample(sample_point_count, return_index=False)

    return SurfacePointCloud(mesh, 
        points=points,
        normals=normals if calculate_normals else None
    )
