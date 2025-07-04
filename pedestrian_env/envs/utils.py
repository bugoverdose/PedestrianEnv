def is_overlapping(left1, right1, left2, right2):
    if right1 < left2 or right2 < left1:
        return False
    return True
