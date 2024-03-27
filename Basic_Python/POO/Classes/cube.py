class Cube:
    def __init__(self, side=1) -> None:
        self.side = side

    # This decorator is used to define a method that operates on the class itself rather than on instances of the class.
    @classmethod
    def create_cube_by_area(cls, area):
        side = area ** 0.5
        return cls(side)
    
    # This decorator is used to define a method that does not access the instance or the class itself. It behaves like a regular function, except that it is defined inside a class.
    @staticmethod
    def falar_oi():
        print('oi')
    
    @property
    def side(self):
        return self._side
    @side.setter
    def side(self, value):
        if not isinstance(value, int) and not isinstance(value, float) or value <= 0:
            raise ValueError("The Cube's side length needs to make sense")
        else:
            self._side = value

    def calculate_area(self) -> float:
        return self.side ** 2

if __name__ == '__main__':
    q1 = Cube()
    q2 = q1.create_cube_by_area(9) #Cube.create_cube_by_area(9)

    print(f'{q1.side}') # 'getter'
    q1.side = 5 # setter
    print(f'{q1.side}') # 'getter'
    print(q1.calculate_area())

    print(q2.side)
    Cube.falar_oi()
