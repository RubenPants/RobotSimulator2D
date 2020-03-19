"""
intersection_test_cy.py

Test of all the intersection methods.
"""
import unittest

from utils.cy.intersection_cy import *
from utils.cy.line2d_cy import Line2dCy
from utils.cy.vec2d_cy import Vec2dCy


class LineLineIntersectionCy(unittest.TestCase):
    """Test the intersection formula between two lines."""
    
    def test_intersect(self):
        """> Test if the intersection of two lines is determined correctly."""
        l1 = Line2dCy(Vec2dCy(1, 0), Vec2dCy(-1, 0))
        l2 = Line2dCy(Vec2dCy(0, 1), Vec2dCy(0, -1))
        self.assertEqual(line_line_intersection_cy(l1, l2), (True, Vec2dCy(0, 0)))
    
    def test_intersect_small(self):
        """> Test if very small lines still can intersect."""
        l1 = Line2dCy(Vec2dCy(1, 0), Vec2dCy(-1, 0))
        l2 = Line2dCy(Vec2dCy(0, 1e-5), Vec2dCy(0, -1e-5))
        self.assertEqual(line_line_intersection_cy(l1, l2), (True, Vec2dCy(0, 0)))
    
    def test_intersect_cross(self):
        """> Test a more sophisticated intersection."""
        l1 = Line2dCy(Vec2dCy(1, 1), Vec2dCy(-1, -1))
        l2 = Line2dCy(Vec2dCy(1, -1), Vec2dCy(-1, 1))
        self.assertEqual(line_line_intersection_cy(l1, l2), (True, Vec2dCy(0, 0)))
    
    def test_intersect_edge(self):
        """> Test the edge-case of the intersection method."""
        l1 = Line2dCy(Vec2dCy(1, 0), Vec2dCy(-1, 0))
        l2 = Line2dCy(Vec2dCy(0, 1), Vec2dCy(0, 0))
        self.assertEqual(line_line_intersection_cy(l1, l2), (True, Vec2dCy(0, 0)))
    
    def test_no_intersect(self):
        """> Test response when no intersection"""
        l1 = Line2dCy(Vec2dCy(1, 0), Vec2dCy(-1, 0))
        l2 = Line2dCy(Vec2dCy(0, 1), Vec2dCy(0, 0.5))
        self.assertEqual(line_line_intersection_cy(l1, l2), (False, None))


class PointCircleIntersectionCy(unittest.TestCase):
    """Test the intersection formula between a point and a circle."""
    
    def test_intersect(self):
        """> Test point-circle intersection when point in circle."""
        c = Vec2dCy(0, 0)
        r = 1
        p = Vec2dCy(0, 0)
        self.assertEqual(point_circle_intersection_cy(p, c, r), True)
    
    def test_intersect_edge(self):
        """> Test point-circle intersection when point at edge of circle."""
        c = Vec2dCy(0, 0)
        r = 1
        p = Vec2dCy(1, 0)
        self.assertEqual(point_circle_intersection_cy(p, c, r), True)
    
    def test_no_intersect(self):
        """> Test when no intersection between point and circle."""
        c = Vec2dCy(0, 0)
        r = 1
        p = Vec2dCy(2, 0)
        self.assertEqual(point_circle_intersection_cy(p, c, r), False)


class PointLineIntersectionCy(unittest.TestCase):
    """Test the intersection formula between a point and a line."""
    
    def test_intersect(self):
        """> Test intersection when point laying on line."""
        l = Line2dCy(Vec2dCy(1, 0), Vec2dCy(-1, 0))
        p = Vec2dCy(0, 0)
        self.assertEqual(point_line_intersection_cy(p, l), True)
    
    def test_intersect_edge(self):
        """> Test intersection when point is edge-point of line."""
        l = Line2dCy(Vec2dCy(1, 0), Vec2dCy(-1, 0))
        p = Vec2dCy(1, 0)
        self.assertEqual(point_line_intersection_cy(p, l), True)
    
    def test_no_intersect(self):
        """> Test when point does not lay on line."""
        l = Line2dCy(Vec2dCy(1, 0), Vec2dCy(-1, 0))
        p = Vec2dCy(0, 1)
        self.assertEqual(point_line_intersection_cy(p, l), False)


class CircleLineIntersectionCy(unittest.TestCase):
    """Test the intersection formula between a circle and a line."""
    
    def test_intersect(self):
        """> Test intersection when line goes through circle's center."""
        c = Vec2dCy(0, 0.5)
        r = 1
        l = Line2dCy(Vec2dCy(1, 0), Vec2dCy(-1, 0))
        self.assertEqual(circle_line_intersection_cy(c, r, l), (True, Vec2dCy(0, 0)))
    
    def test_intersect_edge(self):
        """> Test intersection when line lays on the circle's edge."""
        c = Vec2dCy(0, 1)
        r = 1
        l = Line2dCy(Vec2dCy(1, 0), Vec2dCy(-1, 0))
        self.assertEqual(circle_line_intersection_cy(c, r, l), (True, Vec2dCy(0, 0)))
    
    def test_no_intersect(self):
        """> Test when circle does not intersect with line."""
        c = Vec2dCy(0, 2)
        r = 1
        l = Line2dCy(Vec2dCy(1, 0), Vec2dCy(-1, 0))
        self.assertEqual(circle_line_intersection_cy(c, r, l), (False, None))


def main():
    # Test line line intersections
    lli = LineLineIntersectionCy()
    lli.test_intersect()
    lli.test_intersect_cross()
    lli.test_intersect_edge()
    lli.test_intersect_small()
    lli.test_no_intersect()
    
    # Test point circle intersections
    pci = PointCircleIntersectionCy()
    pci.test_intersect()
    pci.test_intersect_edge()
    pci.test_no_intersect()
    
    # Test point line intersections
    pli = PointLineIntersectionCy()
    pli.test_intersect()
    pli.test_intersect_edge()
    pli.test_no_intersect()
    
    # Test circle line intersections
    cli = CircleLineIntersectionCy()
    cli.test_intersect()
    cli.test_intersect_edge()
    cli.test_no_intersect()


if __name__ == '__main__':
    unittest.main()
