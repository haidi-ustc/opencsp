from opencsp.operations.mutation.crystal import CrystalMutation
from opencsp.operations.crossover.crystal import CrystalCrossover
from opencsp.adapters.registry import OperationRegistry
opr=OperationRegistry()
opr.register_crossover(CrystalCrossover(),3)
opr.register_mutation(CrystalMutation(),3)

