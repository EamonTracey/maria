from enum import StrEnum

class Move(StrEnum):
    RIGHT = "right"
    LEFT = "left"
    DOWN = "down"
    UP = "up"
    START = "start"
    SELECT = "select"
    B = "B"
    A = "A"
    NOOP = "NOOP"
