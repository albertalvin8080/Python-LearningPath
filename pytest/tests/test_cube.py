import pytest
import source.geometric_form as form

# pytest tests/test_cube.py

class TestCube:
    def setup_method(self):
        self.cube = form.Cube(5, 6, 7)

    def teardown_method(self):
        # del self.cube # It will be replaced anyway.
        pass 

    def test_area(self):
        assert self.cube.surface_area() == 36

    def test_volume(self):
        assert self.cube.volume() == 210