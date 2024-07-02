import unittest
from unittest.mock import patch
from product import Product


class TestProduct(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        print("Inside setUpClass")

    @classmethod
    def tearDownClass(cls) -> None:
        print("Inside tearDownClass")

    def setUp(self):
        self.product = Product(99, "TV", 200.99, 4)

    def tearDown(self):
        print("Inside tear down")

    def test_total_pricing(self):
        tp = self.product.total_pricing
        self.assertEqual(tp, 200.99 * 4)

    def test_update_db_context_manager(self):
        # product.update_product is the function OUTSIDE the Product class.
        with patch("product.update_product") as mock_update_product:
            mock_update_product.return_value.ok = True
            mock_update_product.return_value.text = "Product updated successfully."
            self.assertEqual(self.product.update_db(), "Product updated successfully.")

            mock_update_product.return_value.ok = False
            mock_update_product.return_value.text = "Failed to update product."
            self.assertEqual(self.product.update_db(), "Failed to update product.")

            mock_update_product.side_effect = ValueError("An error occurred.")
            self.assertEqual(
                self.product.update_db(),
                "Failed to update product. Check the documentation.",
            )

    # This does exactly the same test as the method above, but
    # using patch as a decorator instead of as a context manager.
    @patch("product.update_product")
    def test_update_db_decorator(self, mock_update_product):
        mock_update_product.return_value.ok = True
        mock_update_product.return_value.text = "Product updated successfully."
        self.assertEqual(self.product.update_db(), "Product updated successfully.")

        mock_update_product.return_value.ok = False
        mock_update_product.return_value.text = "Failed to update product."
        self.assertEqual(self.product.update_db(), "Failed to update product.")

        mock_update_product.side_effect = ValueError("An error occurred.")
        self.assertEqual(
            self.product.update_db(),
            "Failed to update product. Check the documentation.",
        )


if __name__ == "__main__":
    unittest.main()
