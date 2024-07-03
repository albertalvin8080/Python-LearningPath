import pytest
import source.calculator as calc

# pytest tests/test_calculator.py

def test_add():
    assert calc.add(1, 4) == 5
    assert calc.add(0, -1) == -1
    assert calc.add(7.24, 83.2) == 90.44

def test_sub():
    assert calc.sub(0, 0) == 0
    assert calc.sub(0, -10) == 10
    assert calc.sub(0, 120) == -120

def test_div():
    assert calc.div(10, 5) == 2

    # Error due to floating point precision
    # assert calc.div(10, -9) == -1.11 
    # assert calc.div(47, 2.3) == 20.434

    assert calc.div(10, -9) == pytest.approx(-1.11, rel=1e-2) # Checks 2 decimal places
    assert calc.div(47, 2.3) == pytest.approx(20.434, rel=1e-3) # Checks 3 decimal places

    # Testing exception
    with pytest.raises(ZeroDivisionError):
        calc.div(0, 0)

