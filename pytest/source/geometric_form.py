class GeometricForm:
    def volume(self) -> float:
        raise NotImplementedError("method volume() not implemented.")

    def surface_area(self) -> float:
        raise NotImplementedError("method arsurface_area() not implemented.")


class Cube(GeometricForm):
    def __init__(self, length, width, height) -> None:
        self.length = length
        self.width = width
        self.height = height

    def volume(self):
        return self.length * self.width * self.height

    def surface_area(self) -> float:
        # return 2 * self.length + 2 * self.width + 2 * self.height
        return 2 * (self.length + self.width + self.height)
