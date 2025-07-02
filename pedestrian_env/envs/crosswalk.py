import math

class CrossWalk:
    def __init__(self, row1, row2, col):
        self.row1 = row1
        self.row2 = row2
        self.col = col

# first-class object for simplicity
class CrossWalks:
    def __init__(self, elements):
        self.elements = elements
        self.active_elements = []

    def update_activation(self, agent):
        agent_x, agent_y, agent_radius = agent.get_cur_pos()
        self.elements = []
        for crosswalk in self.elements:
            [row1, row2, col] = crosswalk
            start_y = row1 - 1
            end_y = row2 + 1
            x = col
            if start_y <= agent_y <= end_y:
                distance = abs(agent_x - x) # check only row distance if in the same danger zone
            else:
                dy = min(abs(agent_y - start_y), abs(agent_y - end_y))
                dx = abs(agent_x - x)
                distance = math.hypot(dx, dy)
            if distance <= agent_radius:
                self.active_elements.append(crosswalk)
