# opencsp/core/structure_generator.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union

class StructureGenerator(ABC):
    """结构生成器基类"""
    
    def __init__(self, composition: Dict[str, int], constraints: Optional[List[Any]] = None):
        """
        初始化结构生成器
        
        Args:
            composition: 化学成分，格式为 {元素符号: 原子数量}
            constraints: 结构约束列表
        """
        self.composition = composition
        self.constraints = constraints or []
        
    @abstractmethod
    def generate(self, n: int = 1) -> List[Any]:
        """
        生成n个结构
        
        Args:
            n: 要生成的结构数量
            
        Returns:
            结构列表
        """
        pass
        
    def is_valid(self, structure: Any) -> bool:
        """检查结构是否满足所有约束"""
        for constraint in self.constraints:
            if not constraint.is_satisfied(structure):
                return False
        return True


class RandomStructureGenerator(StructureGenerator):
    """随机结构生成器"""
    
    def __init__(self, composition: Dict[str, int], volume_range: Tuple[float, float], 
                 dimensionality: int = 3, **kwargs):
        """
        初始化随机结构生成器
        
        Args:
            composition: 化学成分
            volume_range: 体积范围 (min, max)
            dimensionality: 结构维度（1, 2, 3）
            **kwargs: 其他参数
        """
        super().__init__(composition, kwargs.get('constraints'))
        self.volume_range = volume_range
        self.dimensionality = dimensionality
        self.min_distance = kwargs.get('min_distance', 1.5)  # 默认最小原子距离
        
    def generate(self, n: int = 1) -> List[Any]:
        """生成n个随机结构"""
        structures = []
        attempts = 0
        max_attempts = n * 100  # 最大尝试次数
        failed_attempts = 0
        
        print(f"Attempting to generate {n} structures...")
        print(f"Parameters: composition={self.composition}, volume_range={self.volume_range}, dimensionality={self.dimensionality}, min_distance={self.min_distance}")
        
        while len(structures) < n and attempts < max_attempts:
            attempts += 1
            
            # 根据维度生成结构
            if self.dimensionality == 1:
                structure = self._generate_1d()
            elif self.dimensionality == 2:
                structure = self._generate_2d()
            else:  # 3D
                structure = self._generate_3d()
                
            # 检查结构有效性
            if structure is None:
                failed_attempts += 1
                if failed_attempts % 10 == 0:
                    print(f"Failed to generate structure {failed_attempts} times due to atomic distances being too close")
                continue
                
            if self.is_valid(structure):
                structures.append(structure)
                print(f"Successfully generated structure {len(structures)}/{n}")
            else:
                failed_attempts += 1
                if failed_attempts % 10 == 0:
                    print(f"Failed to validate structure {failed_attempts} times")
        
        print(f"Generated {len(structures)}/{n} structures in {attempts} attempts")
        if len(structures) == 0:
            print("WARNING: Failed to generate any valid structures. Try adjusting parameters:")
            print("- Increase volume_range")
            print("- Decrease min_distance")
            print("- Simplify composition")
        
        return structures
        
    def _generate_1d(self) -> Any:
        """生成1D结构或团簇"""
        # 实现生成1D结构的代码
        # 这需要使用pymatgen或ASE来生成实际结构
        # 简单的实现示例:
        try:
            from pymatgen.core import Lattice, Structure
            import numpy as np
            import random
            
            # 计算总原子数
            num_atoms = sum(self.composition.values())
            
            # 创建一个足够大的长方体，使得原子分布在其中
            volume = random.uniform(*self.volume_range)
            # 对于团簇，我们使用一个扁平的盒子
            a = (volume)**(1/3)
            lattice = Lattice.from_parameters(a*3, a*3, a*3, 90, 90, 90)
            
            # 准备原子列表
            species = []
            for element, count in self.composition.items():
                species.extend([element] * count)
                
            # 随机放置原子，集中在盒子中心附近
            coords = []
            for _ in range(num_atoms):
                # 在盒子中心附近随机放置
                x = 0.5 + (random.random() - 0.5) * 0.5
                y = 0.5 + (random.random() - 0.5) * 0.5
                z = 0.5 + (random.random() - 0.5) * 0.5
                coords.append([x, y, z])
                
            # 创建结构
            structure = Structure(lattice, species, coords)
            
            return structure
            
        except ImportError:
            raise ImportError("pymatgen must be installed to generate structures")
        
    def _generate_2d(self) -> Any:
        """生成2D结构"""
        # 实现生成2D结构的代码
        # 类似于_generate_1d，但需要考虑2D的周期性
        # ...
        
    def _generate_3d(self) -> Any:
        """生成3D结构"""
        try:
            from pymatgen.core import Structure, Lattice
            import numpy as np
            
            # 选择随机体积
            volume = np.random.uniform(*self.volume_range)
            
            # 为了团簇结构，使用立方晶胞
            a = (volume) ** (1/3)
            lattice = Lattice.cubic(a)
            
            # 将组分转换为列表
            symbols = []
            for element, count in self.composition.items():
                symbols.extend([element] * count)
            
            num_atoms = len(symbols)
            if num_atoms == 0:
                print("Error: No atoms specified in composition")
                return None
                
            # 尝试不同策略来放置原子
            # 策略1: 完全随机放置在中心附近
            coords = []
            for _ in range(num_atoms):
                x = 0.5 + (np.random.random() - 0.5) * 0.8  # 更大的分散区域
                y = 0.5 + (np.random.random() - 0.5) * 0.8
                z = 0.5 + (np.random.random() - 0.5) * 0.8
                coords.append([x, y, z])
                
            # 检查原子间距是否满足最小距离要求
            too_close = True
            max_attempts = 200  # 增加尝试次数
            attempts = 0
            
            while too_close and attempts < max_attempts:
                too_close = False
                for i in range(len(coords)):
                    for j in range(i+1, len(coords)):
                        # 计算分数坐标下的距离
                        dist = np.linalg.norm(
                            lattice.get_cartesian_coords(coords[i]) - 
                            lattice.get_cartesian_coords(coords[j])
                        )
                        if dist < self.min_distance:
                            # 如果距离太近，重新生成坐标
                            too_close = True
                            # 更随机化坐标位置
                            x = 0.5 + (np.random.random() - 0.5) * 0.8
                            y = 0.5 + (np.random.random() - 0.5) * 0.8
                            z = 0.5 + (np.random.random() - 0.5) * 0.8
                            coords[i] = [x, y, z]
                            break
                    if too_close:
                        break
                attempts += 1
                
            if too_close:
                # 尝试策略2: 在球面上均匀放置原子
                print("Trying spherical placement strategy...")
                coords = []
                radius = (3 * volume / (4 * np.pi)) ** (1/3) * 0.4  # 球半径
                
                for i in range(num_atoms):
                    # 球坐标随机点
                    phi = np.random.random() * 2 * np.pi
                    costheta = np.random.random() * 2 - 1
                    theta = np.arccos(costheta)
                    
                    # 转换为笛卡尔坐标，并放置在单位晶胞的中心
                    r = radius * (0.8 + 0.4 * np.random.random())  # 变化半径以避免完全同心
                    x = 0.5 + r * np.sin(theta) * np.cos(phi) / a
                    y = 0.5 + r * np.sin(theta) * np.sin(phi) / a
                    z = 0.5 + r * np.cos(theta) / a
                    coords.append([x, y, z])
                    
                # 再次检查原子间距
                too_close = False
                for i in range(len(coords)):
                    for j in range(i+1, len(coords)):
                        dist = np.linalg.norm(
                            lattice.get_cartesian_coords(coords[i]) - 
                            lattice.get_cartesian_coords(coords[j])
                        )
                        if dist < self.min_distance:
                            too_close = True
                            break
                    if too_close:
                        break
                
                if too_close:
                    print(f"Unable to place atoms with min_distance={self.min_distance}. Try reducing min_distance or increasing volume_range.")
                    return None
                
            # 创建结构
            structure = Structure(lattice, symbols, coords)
            return structure
            
        except ImportError as e:
            print(f"Error importing required modules: {e}")
            raise ImportError("pymatgen must be installed to generate 3D structures")
        except Exception as e:
            print(f"Error generating 3D structure: {e}")
            return None


class SymmetryBasedStructureGenerator(StructureGenerator):
    """基于对称性的结构生成器"""
    
    def __init__(self, composition: Dict[str, int], spacegroup: Optional[int] = None, 
                 lattice_vectors: Optional[List[List[float]]] = None, dimensionality: int = 3, **kwargs):
        """
        初始化基于对称性的结构生成器
        
        Args:
            composition: 化学成分
            spacegroup: 空间群号（对于3D结构）
            lattice_vectors: 晶格向量
            dimensionality: 结构维度
            **kwargs: 其他参数
        """
        super().__init__(composition, kwargs.get('constraints'))
        self.spacegroup = spacegroup
        self.lattice_vectors = lattice_vectors
        self.dimensionality = dimensionality
        
    def generate(self, n: int = 1) -> List[Any]:
        """生成n个基于对称性的结构"""
        # 实现基于对称性生成结构的代码
        # ...
