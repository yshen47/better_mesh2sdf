import better_mesh2sdf
import numpy as np
import trimesh
import skimage
import torch

if __name__ == '__main__':
    mesh = trimesh.load('sub_world_mesh.obj')
    outside_guidance_point = mesh.center_mass
    outside_guidance_point[1] += 10
    query_points, queried_sdfs = better_mesh2sdf.sample_sdf_near_surface(mesh, outside_guidance_point, number_of_points=100000)

    pcd = trimesh.PointCloud(query_points, np.array(torch.tanh(torch.from_numpy(queried_sdfs.repeat(3, 1)))))
    pcd.show()