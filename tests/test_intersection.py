"""
Test of all the intersection methods.
"""

import unittest

from utils.intersection import *
from utils.line2d import Line2d
from utils.vec2d import Vec2d


class Line2dLine2dIntersection(unittest.TestCase):
    """
    Test the intersection formula between two lines.
    """
    
    def test_intersect(self):
        l1 = Line2d(Vec2d(1, 0), Vec2d(-1, 0))
        l2 = Line2d(Vec2d(0, 1), Vec2d(0, -1))
        self.assertEqual(line_line_intersection(l1, l2), (True, Vec2d(0, 0)))
    
    def test_intersect_small(self):
        l1 = Line2d(Vec2d(1, 0), Vec2d(-1, 0))
        l2 = Line2d(Vec2d(0, 1e-5), Vec2d(0, -1e-5))
        self.assertEqual(line_line_intersection(l1, l2), (True, Vec2d(0, 0)))
    
    def test_intersect_cross(self):
        l1 = Line2d(Vec2d(1, 1), Vec2d(-1, -1))
        l2 = Line2d(Vec2d(1, -1), Vec2d(-1, 1))
        self.assertEqual(line_line_intersection(l1, l2), (True, Vec2d(0, 0)))
    
    def test_intersect_edge(self):
        l1 = Line2d(Vec2d(1, 0), Vec2d(-1, 0))
        l2 = Line2d(Vec2d(0, 1), Vec2d(0, 0))
        self.assertEqual(line_line_intersection(l1, l2), (True, Vec2d(0, 0)))
    
    def test_no_intersect(self):
        l1 = Line2d(Vec2d(1, 0), Vec2d(-1, 0))
        l2 = Line2d(Vec2d(0, 1), Vec2d(0, 0.5))
        self.assertEqual(line_line_intersection(l1, l2), (False, None))


class PointCircleIntersection(unittest.TestCase):
    """
    Test the intersection formula between a point and a circle.
    """
    
    def test_intersect(self):
        c = Vec2d(0, 0)
        r = 1
        p = Vec2d(0, 0)
        self.assertEqual(point_circle_intersection(p, c, r), True)
    
    def test_intersect_edge(self):
        c = Vec2d(0, 0)
        r = 1
        p = Vec2d(1, 0)
        self.assertEqual(point_circle_intersection(p, c, r), True)
    
    def test_no_intersect(self):
        c = Vec2d(0, 0)
        r = 1
        p = Vec2d(2, 0)
        self.assertEqual(point_circle_intersection(p, c, r), False)


class PointLine2dIntersection(unittest.TestCase):
    """
    Test the intersection formula between a point and a line.
    """
    
    def test_intersect(self):
        l = Line2d(Vec2d(1, 0), Vec2d(-1, 0))
        p = Vec2d(0, 0)
        self.assertEqual(point_line_intersection(p, l), True)
    
    def test_intersect_edge(self):
        l = Line2d(Vec2d(1, 0), Vec2d(-1, 0))
        p = Vec2d(1, 0)
        self.assertEqual(point_line_intersection(p, l), True)
    
    def test_no_intersect(self):
        l = Line2d(Vec2d(1, 0), Vec2d(-1, 0))
        p = Vec2d(0, 1)
        self.assertEqual(point_line_intersection(p, l), False)


class CircleLine2dIntersection(unittest.TestCase):
    """
    Test the intersection formula between a circle and a line.
    """
    
    def test_intersect(self):
        c = Vec2d(0, 0.5)
        r = 1
        l = Line2d(Vec2d(1, 0), Vec2d(-1, 0))
        self.assertEqual(circle_line_intersection(c, r, l), (True, Vec2d(0, 0)))
    
    def test_intersect_edge(self):
        c = Vec2d(0, 1)
        r = 1
        l = Line2d(Vec2d(1, 0), Vec2d(-1, 0))
        self.assertEqual(circle_line_intersection(c, r, l), (True, Vec2d(0, 0)))
    
    def test_no_intersect(self):
        c = Vec2d(0, 2)
        r = 1
        l = Line2d(Vec2d(1, 0), Vec2d(-1, 0))
        self.assertEqual(circle_line_intersection(c, r, l), (False, None))


if __name__ == '__main__':
    unittest.main()
