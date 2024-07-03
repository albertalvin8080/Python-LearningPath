import source.geometric_form as form
import pytest

# pytest tests/test_cube.py


def test_volume(my_cube):
    assert my_cube.volume() == 64


def test_surface_area(my_cube):
    assert my_cube.surface_area() == 24


def test_not_equals(my_cube, other_cube):
    assert my_cube != other_cube


@pytest.mark.skip(reason="Not necessary.")
def test_equals(my_cube):
    equal_cube = form.Cube(4)
    assert my_cube == equal_cube


# Tells pytest that this test should fail.
@pytest.mark.xfail(reason="Not implemented.")
def test_euler_formula(my_cube):
    assert my_cube.faces + my_cube.vertices - my_cube.edges == 2
