from enum import Enum, IntEnum

class World(IntEnum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8


class Stage(IntEnum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4


class Move(str, Enum):
    RIGHT = "right"
    LEFT = "left"
    DOWN = "down"
    UP = "up"
    START = "start"
    SELECT = "select"
    B = "B"
    A = "A"
    NOOP = "NOOP"
