"""
Test of all the intersection methods.
"""
import unittest

from utils.intersection import *
from utils.line2d import Line2d
from utils.vec2d import Vec2d


class LineLineIntersection(unittest.TestCase):
    """Test the intersection formula between two lines."""
    
    def test_intersect(self):
        """> Test if the intersection of two lines is determined correctly."""
        l1 = Line2d(Vec2d(1, 0), Vec2d(-1, 0))
        l2 = Line2d(Vec2d(0, 1), Vec2d(0, -1))
        self.assertEqual(line_line_intersection(l1, l2), (True, Vec2d(0, 0)))
    
    def test_intersect_small(self):
        """> Test if very small lines still can intersect."""
        l1 = Line2d(Vec2d(1, 0), Vec2d(-1, 0))
        l2 = Line2d(Vec2d(0, 1e-5), Vec2d(0, -1e-5))
        self.assertEqual(line_line_intersection(l1, l2), (True, Vec2d(0, 0)))
    
    def test_intersect_cross(self):
        """> Test a more sophisticated intersection."""
        l1 = Line2d(Vec2d(1, 1), Vec2d(-1, -1))
        l2 = Line2d(Vec2d(1, -1), Vec2d(-1, 1))
        self.assertEqual(line_line_intersection(l1, l2), (True, Vec2d(0, 0)))
    
    def test_intersect_edge(self):
        """> Test the edge-case of the intersection method."""
        l1 = Line2d(Vec2d(1, 0), Vec2d(-1, 0))
        l2 = Line2d(Vec2d(0, 1), Vec2d(0, 0))
        self.assertEqual(line_line_intersection(l1, l2), (True, Vec2d(0, 0)))
    
    def test_no_intersect(self):
        """> Test response when no intersection"""
        l1 = Line2d(Vec2d(1, 0), Vec2d(-1, 0))
        l2 = Line2d(Vec2d(0, 1), Vec2d(0, 0.5))
        self.assertEqual(line_line_intersection(l1, l2), (False, None))


class PointCircleIntersection(unittest.TestCase):
    """Test the intersection formula between a point and a circle."""
    
    def test_intersect(self):
        """> Test point-circle intersection when point in circle."""
        c = Vec2d(0, 0)
        r = 1
        p = Vec2d(0, 0)
        self.assertEqual(point_circle_intersection(p, c, r), True)
    
    def test_intersect_edge(self):
        """> Test point-circle intersection when point at edge of circle."""
        c = Vec2d(0, 0)
        r = 1
        p = Vec2d(1, 0)
        self.assertEqual(point_circle_intersection(p, c, r), True)
    
    def test_no_intersect(self):
        """> Test when no intersection between point and circle."""
        c = Vec2d(0, 0)
        r = 1
        p = Vec2d(2, 0)
        self.assertEqual(point_circle_intersection(p, c, r), False)


class PointLineIntersection(unittest.TestCase):
    """Test the intersection formula between a point and a line."""
    
    def test_intersect(self):
        """> Test intersection when point laying on line."""
        l = Line2d(Vec2d(1, 0), Vec2d(-1, 0))
        p = Vec2d(0, 0)
        self.assertEqual(point_line_intersection(p, l), True)
    
    def test_intersect_edge(self):
        """> Test intersection when point is edge-point of line."""
        l = Line2d(Vec2d(1, 0), Vec2d(-1, 0))
        p = Vec2d(1, 0)
        self.assertEqual(point_line_intersection(p, l), True)
    
    def test_no_intersect(self):
        """> Test when point does not lay on line."""
        l = Line2d(Vec2d(1, 0), Vec2d(-1, 0))
        p = Vec2d(0, 1)
        self.assertEqual(point_line_intersection(p, l), False)


class CircleLineIntersection(unittest.TestCase):
    """Test the intersection formula between a circle and a line."""
    
    def test_intersect(self):
        """> Test intersection when line goes through circle's center."""
        c = Vec2d(0, 0.5)
        r = 1
        l = Line2d(Vec2d(1, 0), Vec2d(-1, 0))
        self.assertEqual(circle_line_intersection(c, r, l), (True, Vec2d(0, 0)))
    
    def test_intersect_edge(self):
        """> Test intersection when line lays on the circle's edge."""
        c = Vec2d(0, 1)
        r = 1
        l = Line2d(Vec2d(1, 0), Vec2d(-1, 0))
        self.assertEqual(circle_line_intersection(c, r, l), (True, Vec2d(0, 0)))
    
    def test_no_intersect(self):
        """> Test when circle does not intersect with line."""
        c = Vec2d(0, 2)
        r = 1
        l = Line2d(Vec2d(1, 0), Vec2d(-1, 0))
        self.assertEqual(circle_line_intersection(c, r, l), (False, None))


def main():
    # Test line line intersections
    lli = LineLineIntersection()
    lli.test_intersect()
    lli.test_intersect_cross()
    lli.test_intersect_edge()
    lli.test_intersect_small()
    lli.test_no_intersect()
    
    # Test point circle intersections
    pci = PointCircleIntersection()
    pci.test_intersect()
    pci.test_intersect_edge()
    pci.test_no_intersect()
    
    # Test point line intersections
    pli = PointLineIntersection()
    pli.test_intersect()
    pli.test_intersect_edge()
    pli.test_no_intersect()
    
    # Test circle line intersections
    cli = CircleLineIntersection()
    cli.test_intersect()
    cli.test_intersect_edge()
    cli.test_no_intersect()


if __name__ == '__main__':
    unittest.main()
