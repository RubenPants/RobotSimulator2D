from utils.line2d import Line2d
from utils.vec2d import Vec2d

x = Vec2d(1, 0)
y = Vec2d(-1, 0)
line = Line2d(x, y)
print(line.get_length())
