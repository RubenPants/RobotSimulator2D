import math

from utils.vec2d import Vec2d


class Line2d(object):
    def __init__(self, x: Vec2d = None, y: Vec2d = None):
        self.x = x if x else Vec2d(0, 0)
        self.y = y if y else Vec2d(0, 0)
    
    def __setitem__(self, i, value):
        if i == 0:
            self.x = value
        elif i == 1:
            self.y = value
        else:
            raise IndexError()
    
    def __eq__(self, other):
        if hasattr(other, "__getitem__") and len(other) == 2:
            return self.x == other[0] and self.y == other[1]
        else:
            return False
    
    def __add__(self, other):
        if isinstance(other, Line2d):
            return Line2d(self.x + other.x, self.y + other.y)
        elif hasattr(other, "__getitem__"):
            return Line2d(self.x + other[0], self.y + other[1])
        else:
            return Line2d(self.x + other, self.y + other)
    
    __radd__ = __add__
    
    def __iadd__(self, other):
        if isinstance(other, Line2d):
            self.x += other.x
            self.y += other.y
        elif hasattr(other, "__getitem__"):
            self.x += other[0]
            self.y += other[1]
        else:
            self.x += other
            self.y += other
        return self
    
    # Subtraction
    def __sub__(self, other):
        if isinstance(other, Line2d):
            return Line2d(self.x - other.x, self.y - other.y)
        elif hasattr(other, "__getitem__"):
            return Line2d(self.x - other[0], self.y - other[1])
        else:
            return Line2d(self.x - other, self.y - other)
    
    def __rsub__(self, other):
        if isinstance(other, Line2d):
            return Line2d(other.x - self.x, other.y - self.y)
        if hasattr(other, "__getitem__"):
            return Line2d(other[0] - self.x, other[1] - self.y)
        else:
            return Line2d(other - self.x, other - self.y)
    
    def __isub__(self, other):
        if isinstance(other, Line2d):
            self.x -= other.x
            self.y -= other.y
        elif hasattr(other, "__getitem__"):
            self.x -= other[0]
            self.y -= other[1]
        else:
            self.x -= other
            self.y -= other
        return self
    
    def get_length(self):
        return math.sqrt((self.x.x - self.y.x) ** 2 + (self.x.y - self.y.y) ** 2)
