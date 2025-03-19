# opencsp/utils/serialization.py
from typing import Any, Dict

def structure_to_dict(structure: Any) -> Dict[str, Any]:
    """
    将结构对象序列化为字典
    
    Args:
        structure: 结构对象（pymatgen结构或ASE原子）
        
    Returns:
        字典表示
    """
    if hasattr(structure, 'as_dict'):  # pymatgen Structure
        return {
            'type': 'pymatgen',
            'data': structure.as_dict()
        }
    elif hasattr(structure, 'get_positions'):  # ASE Atoms
        from ase.io import write
        import io
        
        # 使用ASE的JSON格式序列化
        buffer = io.StringIO()
        write(buffer, structure, format='json')
        return {
            'type': 'ase',
            'data': buffer.getvalue()
        }
    else:
        raise ValueError(f"Unknown structure type: {type(structure)}")


def dict_to_structure(d: Dict[str, Any]) -> Any:
    """
    从字典反序列化结构对象
    
    Args:
        d: 结构的字典表示
        
    Returns:
        结构对象
    """
    if d['type'] == 'pymatgen':
        from pymatgen.core import Structure
        return Structure.from_dict(d['data'])
    elif d['type'] == 'ase':
        from ase.io import read
        import io
        
        buffer = io.StringIO(d['data'])
        return read(buffer, format='json')
    else:
        raise ValueError(f"Unknown structure type: {d['type']}")
