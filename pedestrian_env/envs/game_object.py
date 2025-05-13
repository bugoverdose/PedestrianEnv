import pygame

class GameObject:

    def __init__(self, image_path, x, y, width, height, pix_square_size):
        self.object_id = id(self) % 1000
        object_image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(object_image, (width, height))

        self.x = x
        self.y = y

        self.width = width
        self.height = height
        self.pix_square_size = pix_square_size

    def render(self, background):
        background.blit(self.image, (self.x * self.pix_square_size, self.y * self.pix_square_size))

class Car(GameObject):

    speed = 1

    def __init__(self, x, y, pix_square_size, car_type_seed = 0):
        self.car_type = (car_type_seed % 12)
        super().__init__(f"sprites/cars/car-side-view{self.car_type}.png", x, y, pix_square_size, pix_square_size, pix_square_size)

    def move(self, max_width):
        self.x += self.speed
        if self.x >= max_width:
            self.x = 0

    def __str__(self):
        return f"Car{self.object_id}: type={self.car_type}, cur_pos=({self.x, self.y})"
