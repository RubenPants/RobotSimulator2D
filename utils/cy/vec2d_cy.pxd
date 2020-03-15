"""
vec2d_cy.pxd

Used to declare all the methods of Vec2dCy that must be callable from outside of other objects.
"""

cdef class Vec2dCy:
    cdef public float x, y
    
    cpdef Vec2dCy _operator_handler(self, other, f)
    
    cpdef Vec2dCy _right_operator_handler(self, other, f)
    
    cpdef Vec2dCy _inplace_operator_handler(self, other, f)
    
    cpdef float get_angle(self)
    
    cpdef float get_length(self)
    
    cpdef Vec2dCy normalized(self)
    
    cpdef tuple get_tuple(self)

cpdef Vec2dCy angle_to_vec(float angle)
