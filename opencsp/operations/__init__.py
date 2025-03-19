# opencsp/operations/__init__.py
"""
操作模块包含各种结构操作的基类和针对不同维度系统的具体实现。
"""

from opencsp.operations.base import StructureOperation

# 导入交叉操作
from opencsp.operations.crossover.cluster import ClusterCrossover
from opencsp.operations.crossover.surface import SurfaceCrossover
from opencsp.operations.crossover.crystal import CrystalCrossover

# 导入变异操作
from opencsp.operations.mutation.cluster import ClusterMutation
from opencsp.operations.mutation.surface import SurfaceMutation
from opencsp.operations.mutation.crystal import CrystalMutation

# 导入位置更新操作
from opencsp.operations.position.cluster import ClusterPositionUpdate
from opencsp.operations.position.surface import SurfacePositionUpdate
from opencsp.operations.position.crystal import CrystalPositionUpdate

# 导入速度更新操作
from opencsp.operations.velocity.cluster import ClusterVelocityUpdate, initialize_cluster_velocity
from opencsp.operations.velocity.surface import SurfaceVelocityUpdate, initialize_surface_velocity
from opencsp.operations.velocity.crystal import CrystalVelocityUpdate, initialize_crystal_velocity

__all__ = [
    'StructureOperation',
    'ClusterCrossover',
    'SurfaceCrossover',
    'CrystalCrossover',
    'ClusterMutation',
    'SurfaceMutation',
    'CrystalMutation',
    'ClusterPositionUpdate',
    'SurfacePositionUpdate',
    'CrystalPositionUpdate',
    'ClusterVelocityUpdate',
    'SurfaceVelocityUpdate',
    'CrystalVelocityUpdate',
    'initialize_cluster_velocity',
    'initialize_surface_velocity',
    'initialize_crystal_velocity'
]
