"""
intersection_cy.pxd

Used to declare all the intersection_cy methods that must be callable from outside of other objects.
"""
from utils.cy.line2d_cy cimport Line2dCy
from utils.cy.vec2d_cy cimport Vec2dCy

cpdef tuple line_line_intersection_cy(Line2dCy l1, Line2dCy l2)

cpdef bint point_circle_intersection_cy(Vec2dCy p, Vec2dCy c, float r)

cpdef bint point_line_intersection_cy(Vec2dCy p, Line2dCy l)

cpdef tuple circle_line_intersection_cy(Vec2dCy c, float r, Line2dCy l)
