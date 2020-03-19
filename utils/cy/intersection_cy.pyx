"""
intersection_cy.pyx

Cython version of the intersection.py file. Note that this file co-exists with a .pxd file (needed to import the
intersection methods in other files).
"""
from utils.cy.line2d_cy cimport Line2dCy
from utils.cy.vec2d_cy cimport Vec2dCy

# Constants
cdef float EPSILON = 1e-5

cpdef tuple line_line_intersection_cy(Line2dCy l1, Line2dCy l2):
    """
    Determine if two lines are intersecting with each other and give point of contact if they do.
    
    :param l1: Line2d
    :param l2: Line2d
    :return: Bool, Intersection: Vec2d
    """
    cdef float a_dev
    cdef float a
    cdef float b_dev
    cdef float b
    
    a_dev = ((l2.y.y - l2.x.y) * (l1.y.x - l1.x.x) - (l2.y.x - l2.x.x) * (l1.y.y - l1.x.y))
    a_dev = a_dev if a_dev != 0 else EPSILON
    a = ((l2.y.x - l2.x.x) * (l1.x.y - l2.x.y) - (l2.y.y - l2.x.y) * (l1.x.x - l2.x.x)) / a_dev
    b_dev = ((l2.y.y - l2.x.y) * (l1.y.x - l1.x.x) - (l2.y.x - l2.x.x) * (l1.y.y - l1.x.y))
    b_dev = b_dev if b_dev != 0 else EPSILON
    b = ((l1.y.x - l1.x.x) * (l1.x.y - l2.x.y) - (l1.y.y - l1.x.y) * (l1.x.x - l2.x.x)) / b_dev
    
    # Check if not intersecting
    if 0 <= a <= 1 and 0 <= b <= 1:
        return True, Vec2dCy(l1.x.x + (a * (l1.y.x - l1.x.x)), l1.x.y + (a * (l1.y.y - l1.x.y)))
    else:
        return False, None

cpdef bint point_circle_intersection_cy(Vec2dCy p, Vec2dCy c, float r):
    """
    Determine if a point lays inside of a circle.
    
    :param p: Point
    :param c: Center of circle
    :param r: Radius of circle
    :return: Bool
    """
    return (p - c).get_length() < r + EPSILON

cpdef bint point_line_intersection_cy(Vec2dCy p, Line2dCy l):
    """
    Determine if a point lays on a line.
    
    :param p: Point
    :param l: Line2d
    :return: Bool
    """
    return (l.get_length()) - EPSILON <= ((p - l.x).get_length() + (p - l.y).get_length()) <= (l.get_length() + EPSILON)

cpdef tuple circle_line_intersection_cy(Vec2dCy c, float r, Line2dCy l):
    """
    Determine if a circle intersects with a line and give point of contact if they do.
    
    :param l: Line2d
    :param c: Center of circle
    :param r: Radius of circle
    :return: Bool, Intersection: Vec2d
    """
    cdef float dot
    cdef Vec2dCy closest
    
    # Check for the edges of the line
    if point_circle_intersection_cy(l.x, c, r):
        return True, l.x
    if point_circle_intersection_cy(l.y, c, r):
        return True, l.y
    
    # Determine closest point to the line
    dot = (((c.x - l.x.x) * (l.y.x - l.x.x)) + ((c.y - l.x.y) * (l.y.y - l.x.y))) / (l.get_length() ** 2)
    closest = Vec2dCy(l.x.x + (dot * (l.y.x - l.x.x)), l.x.y + (dot * (l.y.y - l.x.y)))
    
    # Check if closest is on segment
    if not point_line_intersection_cy(p=closest, l=l):
        return False, None
    
    # Check if in circle
    return (True, closest) if (closest - c).get_length() <= (r + EPSILON) else (False, None)
