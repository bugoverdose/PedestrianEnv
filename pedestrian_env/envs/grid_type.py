from enum import Enum

class GridType(Enum):
    CROSSWALK = 0
    SAFE = 1
    DANGER = 2
    UNREACHABLE = 3

    def __str__(self):
        string_map = {
            GridType.CROSSWALK: "CrossWalk",
            GridType.SAFE: "Safe",
            GridType.DANGER: "Danger",
            GridType.UNREACHABLE: "Unreachable",
        }
        return string_map[self]
