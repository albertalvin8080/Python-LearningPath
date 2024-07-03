import pytest
import source.calculator as calc

# pytest tests/test_calculatorV2.py


@pytest.fixture
def a():
    return 999


@pytest.fixture
def b():
    return 997


def test_local_fixture(a, b):
    assert a - b == 2


# The number of times this test will run is determined by the number of tuples passed as argvalues.
@pytest.mark.parametrize(
    argnames="a, b, expected", argvalues=[(5, 6, 11), (9, -1, 8), (-13, 88, 75)]
)
def test_add(a, b, expected):
    assert calc.add(a, b) == expected


@pytest.mark.parametrize(
    "a, b, expected", [(-0, -1, 1), (1, -1, 2), (39, 178892, -178853)]
)
def test_sub(a, b, expected):
    assert calc.sub(a, b) == expected
