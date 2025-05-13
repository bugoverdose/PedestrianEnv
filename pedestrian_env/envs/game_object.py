import pygame

class GameObject:

    def __init__(self, image_path, x, y, width, height):
        self.object_id = id(self) % 1000
        object_image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(object_image, (width, height))

        self.x = x
        self.y = y

        self.width = width
        self.height = height

    def render(self, background):
        background.blit(self.image, (self.x, self.y))

class Car(GameObject):

    speed = 1

    def __init__(self, x, y, pix_square_size, car_type_seed = 0):

        self.car_type = (car_type_seed % 12)
        super().__init__(f"sprites/cars/car-side-view{self.car_type}.png", x*pix_square_size, y*pix_square_size, pix_square_size, pix_square_size)

    def move(self):
        self.x += self.speed

    def __str__(self):
        return f"Car{self.object_id}: type={self.car_type}, cur_pos=({self.x, self.y})"
