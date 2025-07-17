import pygame

from enum import Enum

class CarColorType(Enum):
    GREEN = 0
    YELLOW = 1
    RED = 2

class RiskDetail:
    def __init__(self, color_name, color, penalty):
        self.color_name = color_name
        self.color = color
        self.penalty = penalty

_RiskDetails = {
    CarColorType.RED: RiskDetail(color_name="Red", color=(255, 0, 0), penalty=1000),
    CarColorType.YELLOW: RiskDetail(color_name="Yellow", color=(255, 255, 0), penalty=500),
    CarColorType.GREEN: RiskDetail(color_name="Green", color=(0, 255, 0), penalty=100),
}

 # key=height, value=(car_name, height:width ratio)
CAR_CANDIDATES = {
    1: [
        ("MICRO", 1.1875),
        ("JEEP", 1.3043),
        ("CIVIC", 1.5556),
        ("VAN", 1.6136),
        ("SUV", 1.8333),
        ("MINIVAN", 1.9070),
        ("WAGON", 1.9545),
        ("HatchBack", 1.9750),
        ("SPORT", 1.9756),
        ("COUPE", 2.0238),
        ("MUSCLECAR", 2.0244),
        ("LUXURY", 2.0698),
        ("SEDAN", 2.0750),
        ("SUPERCAR", 2.2895),
        ("LIMO", 3.2250),
    ],
    2: [
        ("CAMPER", 1.8868),
        ("MEDIUMTRUCK", 1.8983),
        ("BOXTRUCK", 2.0600),
        ("PICKUP", 2.1250),
        # ("BUS", 2.2540), # Too white
    ]
}

def get_max_car_grid_width():
    max_width = 0
    for height in CAR_CANDIDATES:
        for candidate in CAR_CANDIDATES[height]:
            max_width = max(max_width, candidate[1] * height)
    return max_width

def get_max_car_grid_height():
    max_height = 0
    for height in CAR_CANDIDATES:
        max_height = max(max_height, height)
    return max_height

class CarDetail:
    def __init__(self, car_name, color_type, go_right):
        risk_detail = _RiskDetails[color_type]
        self.color = risk_detail.color
        self.penalty = risk_detail.penalty

        self.go_right = go_right
        direction = "EAST" if go_right else "WEST"
        self.image_path = f"sprites/cars/{risk_detail.color_name}_{car_name}_CLEAN_{direction}_000_cropped.png"
