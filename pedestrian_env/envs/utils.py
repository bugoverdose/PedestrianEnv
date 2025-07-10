import math

def is_overlapping(left1, right1, left2, right2):
    if right1 < left2 or right2 < left1:
        return False
    return True

def is_overlapping_circle_and_rectangle(circle, rectangle):
    (cx, cy, radius) = circle
    (left_x, right_x, top_y, bottom_y) = rectangle
    closest_x = max(left_x, min(cx, right_x))
    closest_y = max(top_y, min(cy, bottom_y))
    distance = math.sqrt((closest_x - cx) ** 2 + (closest_y - cy) ** 2)
    return distance < radius
