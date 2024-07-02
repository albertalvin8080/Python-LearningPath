import unittest
from calculator import Calculator


class TestCalculator(unittest.TestCase):
    calculator = Calculator()

    def test_add(self):
        localCalc = TestCalculator.calculator
        self.assertEqual(localCalc.add(10, 2), 12)
        self.assertEqual(localCalc.add(0, 0), 0)
        self.assertEqual(localCalc.add(8, -5), 3)
        self.assertEqual(localCalc.add(-3, -9), -12)

    def test_sub(self):
        localCalc = TestCalculator.calculator
        self.assertEqual(localCalc.sub(0, -1), 1)
        self.assertEqual(localCalc.sub(-1, 0), -1)
        self.assertEqual(localCalc.sub(-12, -24), 12)
        self.assertEqual(localCalc.sub(93, 10), 83)
        self.assertEqual(localCalc.sub(93.32, 10), 83.32)

    def test_mult(self):
        localCalc = TestCalculator.calculator
        self.assertEqual(localCalc.mult(0, -1), 0)
        self.assertEqual(localCalc.mult(-1, 0), 0)
        self.assertEqual(localCalc.mult(2, 97), 194)
        self.assertEqual(localCalc.mult(83, 97), 8051)

    def test_div(self):
        localCalc = TestCalculator.calculator
        self.assertEqual(localCalc.div(0, -1), 0)
        self.assertEqual(localCalc.div(23, 10), 2.3)
        self.assertAlmostEqual(localCalc.div(89, 321), 0.277, places=3)

        # self.assertRaises(ValueError, localCalc.div, 1, 0)
        with self.assertRaises(ValueError):
            localCalc.div(1, 0)

    def test_power(self):
        localCalc = TestCalculator.calculator
        self.assertEqual(localCalc.power(0, 0), 1)
        self.assertEqual(localCalc.power(0, 1), 0)
        self.assertEqual(localCalc.power(23, 0), 1)
        self.assertEqual(localCalc.power(23, 2), 529)


if __name__ == "__main__":
    # NOTE: You could aldo do `python -m unittest test_calculator.py`
    unittest.main()
