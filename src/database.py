import sqlite3
import os

# Define path for the database directory and file
db_directory = '../db'
db_file = 'product_data.db'
db_path = os.path.join(db_directory, db_file)

# Create the database directory if it does not exist
if not os.path.exists(db_directory):
    os.makedirs(db_directory)

# Connect to the SQLite database
connection = sqlite3.connect(db_path)

# Function to create tables
def create_tables():
    with connection:
        # Create product table with id, name, price, and class_id starting from 0
        connection.execute("""
        CREATE TABLE IF NOT EXISTS product (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            price REAL NOT NULL,
            class_id INTEGER NOT NULL
        );
        """)

        # Insert a row with id 0, then delete it to start IDs from 0
        connection.execute("INSERT INTO product (id, name, price, class_id) VALUES (0, 'dummy', 0, 0);")
        connection.execute("DELETE FROM product WHERE id = 0;")

# Insert multiple products
def add_multiple_products(products):
    with connection:
        connection.executemany(
            "INSERT INTO product (name, price, class_id) VALUES (?, ?, ?);",
            products
        )

# Fetch all products
def get_all_products():
    cursor = connection.execute("SELECT * FROM product;")
    return cursor.fetchall()

# Close the database connection
def close_connection():
    connection.close()

def get_product_prices():
    connection = sqlite3.connect('../db/product_data.db')
    cursor = connection.cursor()
    cursor.execute("SELECT name, price FROM product")
    product = cursor.fetchall()

    return {name: price for name, price in product}

def get_product_name():
    # Kết nối đến cơ sở dữ liệu
    connection = sqlite3.connect('../db/product_data.db')
    cursor = connection.cursor()

    # Thực thi câu truy vấn để lấy tên sản phẩm
    cursor.execute("SELECT name FROM product")

    # Lấy tất cả các tên sản phẩm
    product_names = cursor.fetchall()

    # Chuyển đổi thành danh sách tên sản phẩm (lọc lấy phần tử đầu tiên của mỗi tuple)
    return [name[0] for name in product_names]

# Main function
# def main():

    # Create tables if they don't exist
    # create_tables()
    #
    # # Dictionary of products with prices in VND and class_id starting from 0
    # product_prices = {
    #     'BUT_CHI_DIXON': (5000, 0),
    #     'BUT_HIGHLIGHT_MNG_TIM': (20000, 1),
    #     'BUT_HIGHLIGHT_RETRO_COLOR': (15000, 2),
    #     'BUT_LONG_SHARPIE_XANH': (18000, 3),
    #     'BUT_NUOC_CS_8623': (8000, 4),
    #     'XOA_NUOC': (25000, 5),
    #     'HO_DOUBLE_8GM': (15000, 6),
    #     'KEP_BUOM_19MM': (20000, 7),
    #     'KEP_BUOM_25MM': (25000, 8),
    #     'NGOI_CHI_MNG_0.5_100PCS': (30000, 9),
    #     'SO_TAY_A6': (30000, 10),
    #     'THUOC_CAMPUS_15CM': (5000, 11),
    #     'THUOC_DO_DO': (4000, 12),
    #     'THUOC_PARABOL': (4000, 13),
    #     'XOA_KEO_CAPYBARA_9566': (40000, 14)
    # }
    #
    # # Create a list of tuples (name, price, class_id) for each product
    # products_to_add = [(name, price, class_id) for name, (price, class_id) in product_prices.items()]
    #
    # # Add multiple products to the database
    # add_multiple_products(products_to_add)
    #
    # # Fetch and print all products
    # products = get_all_products()
    # if products:
    #     print("All products in the database:")
    #     for product in products:
    #         print(product)
    # else:
    #     print("No products found in the database.")
    #
    # # Close the connection
    # close_connection()




# if __name__ == "__main__":
#     main()
