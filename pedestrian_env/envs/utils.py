def is_overlapping(start1, end1, start2, end2):
    if end1 <= start2 or end2 <= start1:
        return False
    return True
