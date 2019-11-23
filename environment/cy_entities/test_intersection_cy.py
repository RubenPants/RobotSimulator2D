import unittest

from environment.cy_entities.god_class_cy import *


class LineLineIntersectionCy(unittest.TestCase):
    """
    Test the intersection formula between two lines.
    """
    
    def test_intersect(self):
        l1 = Line2dCy(Vec2dCy(1, 0), Vec2dCy(-1, 0))
        l2 = Line2dCy(Vec2dCy(0, 1), Vec2dCy(0, -1))
        self.assertEqual(line_line_intersection_cy(l1, l2), (True, Vec2dCy(0, 0)))
    
    def test_intersect_small(self):
        l1 = Line2dCy(Vec2dCy(1, 0), Vec2dCy(-1, 0))
        l2 = Line2dCy(Vec2dCy(0, 1e-5), Vec2dCy(0, -1e-5))
        self.assertEqual(line_line_intersection_cy(l1, l2), (True, Vec2dCy(0, 0)))
    
    def test_intersect_cross(self):
        l1 = Line2dCy(Vec2dCy(1, 1), Vec2dCy(-1, -1))
        l2 = Line2dCy(Vec2dCy(1, -1), Vec2dCy(-1, 1))
        self.assertEqual(line_line_intersection_cy(l1, l2), (True, Vec2dCy(0, 0)))
    
    def test_intersect_edge(self):
        l1 = Line2dCy(Vec2dCy(1, 0), Vec2dCy(-1, 0))
        l2 = Line2dCy(Vec2dCy(0, 1), Vec2dCy(0, 0))
        self.assertEqual(line_line_intersection_cy(l1, l2), (True, Vec2dCy(0, 0)))
    
    def test_no_intersect(self):
        l1 = Line2dCy(Vec2dCy(1, 0), Vec2dCy(-1, 0))
        l2 = Line2dCy(Vec2dCy(0, 1), Vec2dCy(0, 0.5))
        self.assertEqual(line_line_intersection_cy(l1, l2), (False, None))


class PointCircleIntersectionCy(unittest.TestCase):
    """
    Test the intersection formula between a point and a circle.
    """
    
    def test_intersect(self):
        c = Vec2dCy(0, 0)
        r = 1
        p = Vec2dCy(0, 0)
        self.assertEqual(point_circle_intersection_cy(p, c, r), True)
    
    def test_intersect_edge(self):
        c = Vec2dCy(0, 0)
        r = 1
        p = Vec2dCy(1, 0)
        self.assertEqual(point_circle_intersection_cy(p, c, r), True)
    
    def test_no_intersect(self):
        c = Vec2dCy(0, 0)
        r = 1
        p = Vec2dCy(2, 0)
        self.assertEqual(point_circle_intersection_cy(p, c, r), False)


class PointLineIntersectionCy(unittest.TestCase):
    """
    Test the intersection formula between a point and a line.
    """
    
    def test_intersect(self):
        l = Line2dCy(Vec2dCy(1, 0), Vec2dCy(-1, 0))
        p = Vec2dCy(0, 0)
        self.assertEqual(point_line_intersection_cy(p, l), True)
    
    def test_intersect_edge(self):
        l = Line2dCy(Vec2dCy(1, 0), Vec2dCy(-1, 0))
        p = Vec2dCy(1, 0)
        self.assertEqual(point_line_intersection_cy(p, l), True)
    
    def test_no_intersect(self):
        l = Line2dCy(Vec2dCy(1, 0), Vec2dCy(-1, 0))
        p = Vec2dCy(0, 1)
        self.assertEqual(point_line_intersection_cy(p, l), False)


class CircleLineIntersectionCy(unittest.TestCase):
    """
    Test the intersection formula between a circle and a line.
    """
    
    def test_intersect(self):
        c = Vec2dCy(0, 0.5)
        r = 1
        l = Line2dCy(Vec2dCy(1, 0), Vec2dCy(-1, 0))
        self.assertEqual(circle_line_intersection_cy(c, r, l), (True, Vec2dCy(0, 0)))
    
    def test_intersect_edge(self):
        c = Vec2dCy(0, 1)
        r = 1
        l = Line2dCy(Vec2dCy(1, 0), Vec2dCy(-1, 0))
        self.assertEqual(circle_line_intersection_cy(c, r, l), (True, Vec2dCy(0, 0)))
    
    def test_no_intersect(self):
        c = Vec2dCy(0, 2)
        r = 1
        l = Line2dCy(Vec2dCy(1, 0), Vec2dCy(-1, 0))
        self.assertEqual(circle_line_intersection_cy(c, r, l), (False, None))


def main():
    rel_path = "tests/"
    
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
