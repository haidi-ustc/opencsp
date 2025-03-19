# opencsp/utils/structure.py
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
from pymatgen.analysis.dimensionality import get_dimensionality_cheon,get_dimensionality_gorai
from pymatgen.analysis.dimensionality import get_dimensionality_larsen

def get_structure_dimensionality(structure: Any) -> int:
    """
    判断结构的维度
    
    Args:
        structure: pymatgen Structure或ASE Atoms对象
        
    Returns:
        int: 1(团簇), 2(表面/二维), 3(体相晶体)
    """
    # 将输入统一转换为pymatgen Structure对象
    if hasattr(structure, 'get_positions'):  # ASE Atoms
        # 检查是否为非周期结构（团簇）
        if hasattr(structure, 'pbc') and not any(structure.pbc):
            # 非周期性意味着是团簇
            return 1
            
        try:
            from pymatgen.io.ase import AseAtomsAdaptor
            structure = AseAtomsAdaptor.get_structure(structure)
        except ImportError:
            # 如果缺少pymatgen，使用ASE的方法
            return _get_dimensionality_ase(structure)
    
    try:
        # 对于pymatgen结构，首先检查是否为分子（非周期性团簇）
        if hasattr(structure, 'is_ordered') and not structure.is_ordered:
            # 无序结构通常是团簇
            return 1
            
        # 尝试使用pymatgen的功能判断维度
        try:
            
            # 检查结构是否有有效的晶格
            if not hasattr(structure, 'lattice') or structure.lattice.volume < 1e-6:
                # 这可能是个分子或团簇
                return 1
                
            # 判断结构是否为团簇 - 检查原子数量和晶胞大小
            if len(structure) <= 50 and structure.lattice.volume > 1000:
                # 小团簇在大晶胞中
                return 1
            try:    
                 dim = get_dimensionality_larsen(structure)
            except:
                 dim = get_dimensionality_gorai(structure)
            
            # 将pymatgen的维度映射到我们的维度表示
            if dim == 0:  # 孤立原子或小团簇
                return 3
            elif dim == 1:  # 一维链
                return 3
            elif dim == 2:  # 二维表面
                return 3
            else:  # 三维晶体
                return 3
        except Exception as e:
            print(f"Error in dimension analysis: {e}")
            # 如果分析失败，使用启发式方法
            if hasattr(structure, 'lattice'):
                # 检查晶格参数
                abc = structure.lattice.abc
                if abs(abc[2] - 500) < 1:  # 假设z方向非常大表示2D
                    return 2
                if abs(abc[1] - 500) < 1 and abs(abc[2] - 500) < 1:  # 假设y,z方向非常大表示1D
                    return 1
                
                # 检查是否为团簇 - 大晶胞中的少量原子
                if len(structure) <= 50 and structure.lattice.volume > 1000:
                    return 1
                    
                return 3
    except ImportError as e:
        print(f"Error importing pymatgen modules: {e}")
        # 如果没有pymatgen，使用启发式方法
        pass
            
    # 默认情况下，假设是团簇
    return 1
    
def _get_dimensionality_ase(structure: Any) -> int:
    """使用ASE特有的方法判断结构维度"""
    # 检查是否为非周期结构（团簇）
    if hasattr(structure, 'pbc') and not any(structure.pbc):
        return 1
        
    # 检查单元格大小
    if hasattr(structure, 'cell'):
        cell = structure.cell
        if any(abs(length) < 1e-3 for length in cell.lengths()):
            return 1  # 非周期性或团簇
        if cell.lengths()[2] > 20 and cell.lengths()[0] < 10 and cell.lengths()[1] < 10:
            return 2  # 可能是表面
            
        # 检查是否为团簇 - 大晶胞中的少量原子
        if len(structure) <= 50 and np.prod(cell.lengths()) > 1000:
            return 1
            
        return 3
        
    # 默认情况下，假设是团簇
    return 1


def calculate_structure_distance1(structure1: Any, structure2: Any) -> float:
    """
    计算两个结构之间的差异度量
    
    Args:
        structure1: 第一个结构
        structure2: 第二个结构
        
    Returns:
        float: 结构差异度量
    """
    # 这里可以实现不同的结构差异度量
    # 简单版本：计算原子坐标的均方差
    try:
        # 尝试使用pymatgen的结构匹配器
        from pymatgen.analysis.structure_matcher import StructureMatcher, FrameworkComparator
        
        # 确保结构是pymatgen格式
        if hasattr(structure1, 'get_positions'):  # ASE Atoms
            from pymatgen.io.ase import AseAtomsAdaptor
            structure1 = AseAtomsAdaptor.get_structure(structure1)
            
        if hasattr(structure2, 'get_positions'):  # ASE Atoms
            from pymatgen.io.ase import AseAtomsAdaptor
            structure2 = AseAtomsAdaptor.get_structure(structure2)
        
        # 使用框架比较器（忽略原子类型，只比较结构）
        matcher = StructureMatcher(comparator=FrameworkComparator())
        
        # 计算RMS距离
        try:
            rms = matcher.get_rms_dist(structure1, structure2)
            if rms is None:
                return 10.0  # 结构差异很大
            return rms
        except:
            # 如果匹配失败，返回较大的距离
            return 10.0
            
    except ImportError:
        # 如果没有pymatgen，使用简单的欧氏距离
        import numpy as np
        
        # 获取原子位置
        if hasattr(structure1, 'sites'):  # pymatgen Structure
            pos1 = np.array([site.coords for site in structure1.sites])
        else:  # ASE Atoms
            pos1 = structure1.get_positions()
            
        if hasattr(structure2, 'sites'):  # pymatgen Structure
            pos2 = np.array([site.coords for site in structure2.sites])
        else:  # ASE Atoms
            pos2 = structure2.get_positions()
        
        # 如果原子数不同，无法直接比较
        if len(pos1) != len(pos2):
            return 10.0
            
        # 计算欧氏距离
        return np.sqrt(np.sum((pos1 - pos2)**2) / len(pos1))


import numpy as np
from typing import Any

def calculate_structure_distance(structure1: Any, structure2: Any) -> float:
    """
    计算两个结构之间的差异度量

    Args:
        structure1: 第一个结构
        structure2: 第二个结构

    Returns:
        float: 结构差异度量
    """
    try:
        from pymatgen.analysis.structure_matcher import StructureMatcher, FrameworkComparator
        from pymatgen.io.ase import AseAtomsAdaptor

        # 确保结构是 pymatgen 的 Structure 格式
        if hasattr(structure1, 'get_positions'):  # ASE Atoms
            structure1 = AseAtomsAdaptor.get_structure(structure1)
        if hasattr(structure2, 'get_positions'):  # ASE Atoms
            structure2 = AseAtomsAdaptor.get_structure(structure2)

        # 使用框架比较器（忽略原子类型，只比较结构）
        matcher = StructureMatcher(comparator=FrameworkComparator())

        # 计算 RMS 距离
        try:
            rms_tuple = matcher.get_rms_dist(structure1, structure2)
            if rms_tuple is None:
                return 10.0  # 结构差异很大
            return rms_tuple[0]  # 只取 RMS 距离
        except:
            return 10.0  # 匹配失败时返回较大值

    except ImportError:
        # 如果 pymatgen 不可用，使用欧几里得距离
        # 获取原子位置
        if hasattr(structure1, 'sites'):  # pymatgen Structure
            pos1 = np.array([site.coords for site in structure1.sites])
        else:  # ASE Atoms
            pos1 = structure1.get_positions()

        if hasattr(structure2, 'sites'):  # pymatgen Structure
            pos2 = np.array([site.coords for site in structure2.sites])
        else:  # ASE Atoms
            pos2 = structure2.get_positions()

        # 如果原子数不同，返回一个较大的距离
        if len(pos1) != len(pos2):
            return 10.0

        # 计算均方根欧几里得距离
        return np.sqrt(np.sum((pos1 - pos2) ** 2) / len(pos1))

