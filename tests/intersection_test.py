"""
Test of all the intersection methods.
"""
import unittest

from utils.intersection import *
from utils.line2d import Line2d
from utils.vec2d import Vec2d


class LineLineIntersection(unittest.TestCase):
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


class PointLineIntersection(unittest.TestCase):
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


class CircleLineIntersection(unittest.TestCase):
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


def main():
    success, fail = 0, 0
    
    # Test line line intersections
    lli = LineLineIntersection()
    try:
        lli.test_intersect()
        success += 1
    except AssertionError:
        fail += 1
    try:
        lli.test_intersect_cross()
        success += 1
    except AssertionError:
        fail += 1
    try:
        lli.test_intersect_edge()
        success += 1
    except AssertionError:
        fail += 1
    try:
        lli.test_intersect_small()
        success += 1
    except AssertionError:
        fail += 1
    try:
        lli.test_no_intersect()
        success += 1
    except AssertionError:
        fail += 1
    
    # Test point circle intersections
    pci = PointCircleIntersection()
    try:
        pci.test_intersect()
        success += 1
    except AssertionError:
        fail += 1
    try:
        pci.test_intersect_edge()
        success += 1
    except AssertionError:
        fail += 1
    try:
        pci.test_no_intersect()
        success += 1
    except AssertionError:
        fail += 1
    
    # Test point line intersections
    pli = PointLineIntersection()
    try:
        pli.test_intersect()
        success += 1
    except AssertionError:
        fail += 1
    try:
        pli.test_intersect_edge()
        success += 1
    except AssertionError:
        fail += 1
    try:
        pli.test_no_intersect()
        success += 1
    except AssertionError:
        fail += 1
    
    # Test circle line intersections
    cli = CircleLineIntersection()
    try:
        cli.test_intersect()
        success += 1
    except AssertionError:
        fail += 1
    try:
        cli.test_intersect_edge()
        success += 1
    except AssertionError:
        fail += 1
    try:
        cli.test_no_intersect()
        success += 1
    except AssertionError:
        fail += 1
    
    return success, fail


if __name__ == '__main__':
    unittest.main()
