# better_mesh2sdf
A fast and accurate method that convert 3d mesh to SDF

## Introduction of methods
This repository is modified on top of mesh2sdf python API (https://pypi.org/project/mesh-to-sdf/). Despite a good starting point, there are a bunch of issues when using the original mesh2sdf. 

1. For both their sampling and scan based methods, the sign of SDF field is incorrect for many non-watertight objects. A example obj mesh is included in this repository
2. The "scan" surface point cloud sampling procedure significantly slows down the SDF calculation. 

Alternatively, this respository gives a cleaner, faster and more accurate implementation of mesh2SDF. Our solution works for non-watertight and watertight 3D mesh. 
In our solution, we need to first provide the script with a 3D point location, dubbed as guidance point, which we know 100% is outside of the given 3D mesh. And then we shoot rays from the guidance point 
to every SDF query point location, and count the number of ray-mesh intersections between each SDF query points and the guidance points. 


## Requirements
1. Hole-free Mesh with good normal estimation 

Check example.py for details
