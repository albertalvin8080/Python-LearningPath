import source.geometric_form as form
import pytest

# Received as a parameter inside test functions. 
# May also be used inside class methods.
@pytest.fixture
def my_cube():
    return form.Cube(4)

@pytest.fixture
def other_cube():
    return form.Cube(6)