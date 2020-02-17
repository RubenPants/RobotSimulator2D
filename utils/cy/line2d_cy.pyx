"""
line2d_cy.pyx

Cython version of the line2d.py file. Note that this file co-exists with a .pxd file (needed to import Line2dCy in other
files).
"""
from utils.cy.vec2d_cy import Vec2dCy


cdef class Line2dCy:
    """ Create a two dimensional line setup of the connection between two 2D vectors. """
    
    __slots__ = ("x", "y")
    
    def __init__(self, Vec2dCy x=None, Vec2dCy y=None):
        self.x = x if x else Vec2dCy(0, 0)
        self.y = y if y else Vec2dCy(0, 0)
    
    def __getitem__(self, i):
        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        raise IndexError()
    
    def __setitem__(self, int i, Vec2dCy value):
        if i == 0:
            self.x = value
        elif i == 1:
            self.y = value
        else:
            raise IndexError()
    
    def __iter__(self):
        yield self.x
        yield self.y
    
    def __len__(self):
        return 2
    
    def __repr__(self):
        return 'Line2dCy(%s, %s)' % (self.x, self.y)
    
    def __eq__(self, other):
        if hasattr(other, "__getitem__") and len(other) == 2:
            return self.x == other[0] and self.y == other[1]
        else:
            return False
    
    def __ne__(self, other):
        if hasattr(other, "__getitem__") and len(other) == 2:
            return self.x != other[0] or self.y != other[1]
        else:
            return True
    
    def __add__(self, other):
        if isinstance(other, Line2dCy):
            return Line2dCy(self.x + other.x, self.y + other.y)
        elif hasattr(other, "__getitem__"):
            return Line2dCy(self.x + other[0], self.y + other[1])
        else:
            return Line2dCy(self.x + other, self.y + other)
    
    __radd__ = __add__
    
    def __iadd__(self, other):
        if isinstance(other, Line2dCy):
            self.x += other.x
            self.y += other.y
        elif hasattr(other, "__getitem__"):
            self.x += other[0]
            self.y += other[1]
        else:
            self.x += other
            self.y += other
        return self
    
    def __sub__(self, other):
        if isinstance(other, Line2dCy):
            return Line2dCy(self.x - other.x, self.y - other.y)
        elif hasattr(other, "__getitem__"):
            return Line2dCy(self.x - other[0], self.y - other[1])
        else:
            return Line2dCy(self.x - other, self.y - other)
    
    def __rsub__(self, other):
        if isinstance(other, Line2dCy):
            return Line2dCy(other.x - self.x, other.y - self.y)
        if hasattr(other, "__getitem__"):
            return Line2dCy(other[0] - self.x, other[1] - self.y)
        else:
            return Line2dCy(other - self.x, other - self.y)
    
    def __isub__(self, other):
        if isinstance(other, Line2dCy):
            self.x -= other.x
            self.y -= other.y
        elif hasattr(other, "__getitem__"):
            self.x -= other[0]
            self.y -= other[1]
        else:
            self.x -= other
            self.y -= other
        return self
    
    def __round__(self, n=0):
        return Line2dCy(round(self.x, n), round(self.y, n))
    
    cpdef float get_length(self):
        return (self.x - self.y).get_length()
    
    cpdef float get_orientation(self):
        """
        Get the orientation from start to end.
        """
        return (self.x - self.y).get_angle()
