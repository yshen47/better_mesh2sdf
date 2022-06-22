import numpy as np
import better_mesh2sdf.surface_point_cloud
from better_mesh2sdf.utils import scale_to_unit_cube, scale_to_unit_sphere
import trimesh
from collections import Counter


def get_surface_point_cloud(mesh, sample_point_count=100000):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("The mesh parameter must be a trimesh mesh.")

    return surface_point_cloud.sample_from_mesh(mesh, sample_point_count=sample_point_count, calculate_normals=True)


def mesh_to_sdf(mesh, query_points, bounding_radius=None, sample_point_count=10000000, normal_sample_count=11):
    if not isinstance(query_points, np.ndarray):
        raise TypeError('query_points must be a numpy array.')
    if len(query_points.shape) != 2 or query_points.shape[1] != 3:
        raise ValueError('query_points must be of shape N ✕ 3.')

    point_cloud = get_surface_point_cloud(mesh, sample_point_count)

    return point_cloud.get_sdf_in_batches(query_points, use_depth_buffer=False)


def sample_sdf_near_surface(input_mesh, outside_guidance_point, number_of_points=500000, sample_point_count=1000000, return_in_original_scale=True):
    mesh, scale, translation = scale_to_unit_sphere(input_mesh)

    surface_point_cloud = get_surface_point_cloud(mesh, sample_point_count)
    points, sdfs = surface_point_cloud.sample_sdf_near_surface(number_of_points)

    if return_in_original_scale:
        points = points * scale + translation
        sdfs = sdfs * scale

    sdfs = sanity_check_sdfs(input_mesh, sdfs, points, outside_guidance_point)

    return points, sdfs


def sanity_check_sdfs(mesh, sdfs, points, gurantee_point):
    
    ray_manager = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh, False)

    candidates = list(np.arange(len(points)))

    rays_from_gurantee_point_to_intersection_point = points - gurantee_point
    rays_from_gurantee_point_to_intersection_point /= np.linalg.norm(rays_from_gurantee_point_to_intersection_point, axis=1)[:, None]
    _, index_rays, locations = ray_manager.intersects_id(gurantee_point[None, ].repeat(len(points), 0), rays_from_gurantee_point_to_intersection_point, return_locations=True)

    rays_from_candidate_to_intersection_point = locations - points[index_rays]
    intersect_in_between_flags = np.where(np.matmul(rays_from_gurantee_point_to_intersection_point[index_rays][:, None], rays_from_candidate_to_intersection_point[...,None]).squeeze() < 0)[0]
    intersect_in_between_counts = Counter(list(index_rays[intersect_in_between_flags]))

    odd_ind = []
    for (k, v) in intersect_in_between_counts.items():
        if v % 2 == 1:
            odd_ind.append(k)
    neg_sdf_index = list(np.array(candidates)[np.array(odd_ind)])
    sdfs = np.abs(sdfs)
    sdfs[neg_sdf_index] = -sdfs[neg_sdf_index]
    return sdfs