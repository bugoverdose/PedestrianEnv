import random
from enum import Enum, auto

def generate_rows(height, max_safe_consecutive=3):
    roads = [0] # target area is safe
    safe_count = 1

    for _ in range(height-2):
        choices = list([-1, 0, 1])
        if safe_count >= max_safe_consecutive:
            choices.remove(0)

        cur_row_type = random.choice(choices)
        roads.append(cur_row_type)

        if cur_row_type == 0:
            safe_count += 1
        else:
            safe_count = 0

    roads.append(0) # starting zone is safe
    return roads
