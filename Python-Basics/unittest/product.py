class Product:
    def __init__(self, prod_id, name, price, quantity):
        self.id = prod_id
        self.name = name
        self.price = price
        self.quantity = quantity

    @property
    def total_pricing(self):
        return self.price * self.quantity

    def update_db(self):
        try:
            response = update_product(self)
            if response.ok:
                ret = response.text
            else:
                ret = "Failed to update product."
        except ValueError:
            ret = "Failed to update product. Check the documentation."

        return ret


def update_product(product: Product):
    if not product.id:
        raise ValueError("Product must have an id")
    return {"ok": True, "text": "Product updated successfully."}
