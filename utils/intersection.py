from utils.line2d import Line2d
from utils.vec2d import Vec2d

# Constants
EPSILON = 1e-5


def line_line_intersection(l1: Line2d, l2: Line2d):
    """
    Determine if two lines are intersecting with each other and give point of contact if they do.
    
    :param l1: Line2d
    :param l2: Line2d
    :return: Bool, Intersection: Vec2d
    """
    a_dev = ((l2.y.y - l2.x.y) * (l1.y.x - l1.x.x) - (l2.y.x - l2.x.x) * (l1.y.y - l1.x.y))
    a_dev = a_dev if a_dev != 0 else EPSILON
    a = ((l2.y.x - l2.x.x) * (l1.x.y - l2.x.y) - (l2.y.y - l2.x.y) * (l1.x.x - l2.x.x)) / a_dev
    b_dev = ((l2.y.y - l2.x.y) * (l1.y.x - l1.x.x) - (l2.y.x - l2.x.x) * (l1.y.y - l1.x.y))
    b_dev = b_dev if b_dev != 0 else EPSILON
    b = ((l1.y.x - l1.x.x) * (l1.x.y - l2.x.y) - (l1.y.y - l1.x.y) * (l1.x.x - l2.x.x)) / b_dev
    
    # Check if not intersecting
    if 0 <= a <= 1 and 0 <= b <= 1:
        return True, Vec2d(l1.x.x + (a * (l1.y.x - l1.x.x)), l1.x.y + (a * (l1.y.y - l1.x.y)))
    else:
        return False, None


def point_circle_intersection(p: Vec2d, c: Vec2d, r: float):
    """
    Determine if a point lays inside of a circle.
    
    :param p: Point
    :param c: Center of circle
    :param r: Radius of circle
    :return: Bool
    """
    return (p - c).get_length() < r + EPSILON


def point_line_intersection(p: Vec2d, l: Line2d):
    """
    Determine if a point lays on a line.
    
    :param p: Point
    :param l: Line2d
    :return: Bool
    """
    return l.get_length() - EPSILON <= (p - l.x).get_length() + (p - l.y).get_length() <= l.get_length() + EPSILON


def circle_line_intersection(c: Vec2d, r: float, l: Line2d):
    """
    Determine if a circle intersects with a line and give point of contact if they do.
    
    :param l: Line2d
    :param c: Center of circle
    :param r: Radius of circle
    :return: Bool, Intersection: Vec2d
    """
    # Check for the edges of the line
    if point_circle_intersection(l.x, c, r):
        return True, l.x
    if point_circle_intersection(l.y, c, r):
        return True, l.y
    
    # Determine closest point to the line
    dot = (((c.x - l.x.x) * (l.y.x - l.x.x)) + ((c.y - l.x.y) * (l.y.y - l.x.y))) / (l.get_length() ** 2)
    closest = Vec2d(l.x.x + (dot * (l.y.x - l.x.x)), l.x.y + (dot * (l.y.y - l.x.y)))
    
    # Check if closest is on segment
    if not point_line_intersection(p=closest, l=l):
        return False, None
    
    # Check if in circle
    return (True, closest) if (closest - c).get_length() < r + EPSILON else (False, None)
