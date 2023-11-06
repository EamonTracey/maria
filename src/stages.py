from enum import IntEnum

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


class Version(IntEnum):
    STANDARD = 0
    DOWNSAMPLE = 1
    PIXEL = 2
    RECTANGLE = 3
