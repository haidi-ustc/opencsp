"""Some utility methods, e.g., for getting calculators from well-known sources."""

from __future__ import annotations
import numpy as np
import functools
from inspect import isclass
from typing import TYPE_CHECKING, Any

import ase.optimize
from ase.optimize.optimize import Optimizer

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

# Listing of supported universal calculators.
UNIVERSAL_CALCULATORS = (
    "M3GNet",
    "M3GNet-MP-2021.2.8-PES",
    "M3GNet-MP-2021.2.8-DIRECT-PES",
    "CHGNet",
    "MACE",
    "SevenNet",
    "CHGNet2D",
    "DP",
    "mattersim"
)


@functools.lru_cache
def get_universal_calculator(name: str | Calculator, **kwargs: Any) -> Calculator:
    """Helper method to get some well-known **universal** calculators.
    Imports should be inside if statements to ensure that all models are optional dependencies.
    All calculators must be universal, i.e. encompass a wide swath of the periodic table.
    Though matcalc can be used with any MLIP, even custom ones, this function is not meant as
        a list of all MLIPs.

    Args:
        name (str): Name of calculator.
        **kwargs: Passthrough to calculator init.

    Raises:
        ValueError: on unrecognized model name.

    Returns:
        Calculator
    """
    if not isinstance(name, str):  # e.g. already an ase Calculator instance
        return name

    if name.lower().startswith("m3gnet"):
        import matgl
        from matgl.ext.ase import M3GNetCalculator

        # M3GNet is shorthand for latest M3GNet based on DIRECT sampling.
        name = {"m3gnet": "M3GNet-MP-2021.2.8-DIRECT-PES"}.get(name.lower(), name)
        model = matgl.load_model(name)
        kwargs.setdefault("stress_weight", 1 / 160.21766208)
        return M3GNetCalculator(potential=model, **kwargs)

    if name.lower() == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator

        return CHGNetCalculator(**kwargs)

    if name.lower() == "mace":
        from mace.calculators import mace_mp

        return mace_mp(**kwargs)

    if name.lower() == "sevennet":
        from sevenn.sevennet_calculator import SevenNetCalculator

        return SevenNetCalculator(**kwargs)

    if name.lower() == "mattersim":
        from mattersim.forcefield import MatterSimCalculator
        from mattersim.forcefield.potential  import Potential
        model_file = kwargs.get('model_file', None)
        if  model_file:
            potential = Potential.from_checkpoint(model_file)
            return MatterSimCalculator(potential = potential)
        raise ValueError(f"Support the model_file path")


    if name.lower() == "chgnet2d":
        from chgnet.model.dynamics import CHGNetCalculator
        from chgnet.model import CHGNet
        model_file = kwargs.get('model_file', None)
        if  model_file:
            model = CHGNet.from_file(model_file)
            return CHGNetCalculator(model=model,**kwargs)

        raise ValueError(f"Support the model_file path")

    if name.lower() == "dp":
        from deepmd.calculator import DP
        model_file = kwargs.get('model_file', None)
        if  model_file:
            return DP(model=model_file)
        raise ValueError(f"Support the model_file path")

    raise ValueError(f"Unrecognized {name=}, must be one of {UNIVERSAL_CALCULATORS}")


def is_ase_optimizer(key: str | Optimizer) -> bool:
    """Check if key is the name of an ASE optimizer class."""
    if isclass(key) and issubclass(key, Optimizer):
        return True
    if isinstance(key, str):
        return isclass(obj := getattr(ase.optimize, key, None)) and issubclass(obj, Optimizer)
    return False


VALID_OPTIMIZERS = [key for key in dir(ase.optimize) if is_ase_optimizer(key)]


def get_ase_optimizer(optimizer: str | Optimizer) -> Optimizer:
    """Validate optimizer is a valid ASE Optimizer.

    Args:
        optimizer (str | Optimizer): The optimization algorithm.

    Raises:
        ValueError: on unrecognized optimizer name.

    Returns:
        Optimizer: ASE Optimizer class.
    """
    if isclass(optimizer) and issubclass(optimizer, Optimizer):
        return optimizer

    if optimizer not in VALID_OPTIMIZERS:
        raise ValueError(f"Unknown {optimizer=}, must be one of {VALID_OPTIMIZERS}")

    return getattr(ase.optimize, optimizer) if isinstance(optimizer, str) else optimizer

def prettyprint(c,precision=3):
    c=np.array(c)
    fmt="{:>8."+str(precision)+"f} "
    row = c.shape[0]
    col = c.shape[1]
    for i in range(row):
        for j in range(col):
            print(fmt.format(c[i, j]), end=" ")
            if j == (col - 1):
                print(" ")

def pad_string(s: str, length: int = 40, pad_char: str = '-'):
    r'''
    seperate the output by '-'
    '''
    return s.center(length,pad_char)

