from fastapi import FastAPI, HTTPException, Body, Path
from pydantic import BaseModel
from tortoise import Tortoise, fields
from tortoise.models import Model


app = FastAPI()

# PostgreSQL Database Configuration
DB_CONFIG_PRODUCTS = {
    "connections": {
        "default": "postgres://your_postgres_username:your_postgres_password@localhost:5432/products_db",
    },
    "apps": {
        "models": {
            "models": ["__main__"],  # Points to the models module
            "default_connection": "default",
        }
    }
}

# MongoDB Database Configuration
DB_CONFIG_ORDERS = {
    "connections": {
        "default": "mongodb://your_mongodb_username:your_mongodb_password@localhost:27017/orders_db",
    },
    "apps": {
        "models": {
            "models": ["__main__"],  # Points to the models module
            "default_connection": "default",
        }
    }
}


# Sample data to simulate a database
products_db = {}
orders_db = {}
carts_db = {}

# Models
class Product:
    def __init__(self, name, description, price, category, in_stock):
        self.name = name
        self.description = description
        self.price = price
        self.category = category
        self.in_stock = in_stock

class Order:
    def __init__(self, user_id, products):
        self.user_id = user_id
        self.products = products

class Cart:
    def __init__(self, user_id, products):
        self.user_id = user_id
        self.products = products


# Routes
@app.on_event("startup")
async def startup_db():
    await Tortoise.init(config=DB_CONFIG_PRODUCTS)
    await Tortoise.generate_schemas()

    await Tortoise.init(config=DB_CONFIG_ORDERS)
    await Tortoise.generate_schemas()

@app.on_event("shutdown")
async def shutdown_db():
    await Tortoise.close_connections()

# Products Routes
@app.get("/products")
async def get_products(category: str = None, in_stock: bool = None):
    filtered_products = []

    for product_id, product in products_db.items():
        if (category is None or product.category == category) and \
           (in_stock is None or product.in_stock == in_stock):
            filtered_products.append({"id": product_id, **product.__dict__})

    return filtered_products

@app.get("/products/{product_id}")
async def get_product(product_id: int = Path(..., title="The ID of the product to retrieve")):
    if product_id not in products_db:
        raise HTTPException(status_code=404, detail="Product not found")
    
    product = products_db[product_id]
    return {"id": product_id, **product.__dict__}

@app.post("/products")
async def create_product(product: Product = Body(...)):
    product_id = len(products_db) + 1
    products_db[product_id] = product
    return {"id": product_id, **product.__dict__}

@app.put("/products/{product_id}")
async def update_product(product_id: int = Path(..., title="The ID of the product to update"), product_update: Product = Body(...)):
    if product_id not in products_db:
        raise HTTPException(status_code=404, detail="Product not found")

    existing_product = products_db[product_id]
    existing_product.__dict__.update({k: v for k, v in product_update.__dict__.items() if v is not None})
    
    return {"id": product_id, **existing_product.__dict__}

@app.delete("/products/{product_id}")
async def delete_product(product_id: int = Path(..., title="The ID of the product to delete")):
    if product_id not in products_db:
        raise HTTPException(status_code=404, detail="Product not found")

    deleted_product = products_db.pop(product_id)
    return {"message": f"Product with ID {product_id} has been deleted", **deleted_product.__dict__}

# Orders Routes
@app.post("/orders")
async def create_order(order: Order = Body(...)):
    order_id = len(orders_db) + 1
    orders_db[order_id] = order
    return {"order_id": order_id, **order.__dict__}

@app.get("/orders/{user_id}")
async def get_user_orders(user_id: int = Path(..., title="The ID of the user")):
    user_orders = [order for order in orders_db.values() if order.user_id == user_id]
    if not user_orders:
        raise HTTPException(status_code=404, detail=f"No orders found for user {user_id}")

    return [{"order_id": order_id, **order.__dict__} for order_id, order in enumerate(user_orders, start=1)]

# Cart Routes
@app.post("/cart/{user_id}")
async def add_to_cart(user_id: int = Path(..., title="The ID of the user"), item: Product = Body(...)):
    if user_id not in carts_db:
        carts_db[user_id] = Cart(user_id, [])

    cart = carts_db[user_id]
    cart.products.append(item)
    
    return {"user_id": user_id, **cart.__dict__}

@app.get("/cart/{user_id}")
async def get_cart(user_id: int = Path(..., title="The ID of the user")):
    if user_id not in carts_db:
        raise HTTPException(status_code=404, detail=f"No cart found for user {user_id}")

    return {"user_id": user_id, **carts_db[user_id].__dict__}

@app.delete("/cart/{user_id}/item/{product_id}")
async def remove_from_cart(user_id: int = Path(..., title="The ID of the user"), product_id: int = Path(..., title="The ID of the product to remove")):
    if user_id not in carts_db:
        raise HTTPException(status_code=404, detail=f"No cart found for user {user_id}")

    cart = carts_db[user_id]
    cart.products = [product for product in cart.products if product["id"] != product_id]
    
    return {"user_id": user_id, **cart.__dict__}
