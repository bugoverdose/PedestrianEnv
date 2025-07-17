from enum import Enum

class CarColorType(Enum):
    GREEN = 0
    YELLOW = 1
    RED = 2

class RiskDetail:
    def __init__(self, color, penalty):
        self.color = color
        self.penalty = penalty

_RiskDetails = {
    CarColorType.RED: RiskDetail(color=(255, 0, 0), penalty=1000),
    CarColorType.YELLOW: RiskDetail(color=(255, 255, 0), penalty=500),
    CarColorType.GREEN: RiskDetail(color=(0, 255, 0), penalty=100),
}

class CarDetail:
    def __init__(self, color_type):
        risk_detail = _RiskDetails[color_type]
        self.color = risk_detail.color
        self.penalty = risk_detail.penalty
