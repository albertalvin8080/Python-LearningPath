class GeometricForm:
    def volume(self) -> float:
        raise NotImplementedError("method volume() not implemented.")

    def surface_area(self) -> float:
        raise NotImplementedError("method arsurface_area() not implemented.")


class Cuboid(GeometricForm):
    def __init__(self, length, width, height) -> None:
        self.length = length
        self.width = width
        self.height = height

    def volume(self):
        return self.length * self.width * self.height

    def surface_area(self) -> float:
        # return 2 * self.length + 2 * self.width + 2 * self.height
        return 2 * (self.length + self.width + self.height)


class Cube(Cuboid):
    def __init__(self, side) -> None:
        super().__init__(side, side, side)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Cube):
            return False
        return self.surface_area() == other.surface_area() and self.volume() == other.volume()
