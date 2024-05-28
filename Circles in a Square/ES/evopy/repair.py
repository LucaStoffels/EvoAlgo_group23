"""Module containing enum Repair describing different repair techniques available to the algorithm."""
from enum import Enum


class Repair(Enum):
    """Enum used to distinguish different types of repair.

    These repair techniques are used to determine the mechanism which features of 
    each individual are repaired to fall within the bounds. The three repair techniques
    which are included are:

    - RANDOM_REPAIR: Alleles whose values fall outside the given bounds are re-selected
                      uniform randomly.
    - BOUNDARY_REPAIR: Alleles whose values fall outside the given bounds are set
                      to the closest bound.
    - CONSTRAINT_DOMINATION: Alleles whose values fall outside the bounds are still kept
                        but the amount of distance that they fall outside the bounds are 
                        penalised quadratically.
    """
    RANDOM_REPAIR = 1
    BOUNDARY_REPAIR = 2
    CONSTRAINT_DOMINATION = 3