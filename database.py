import os

import psycopg
from dotenv import load_dotenv


load_dotenv()


DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
    return psycopg.connect(DATABASE_URL)


if __name__ == "__main__":
    connection = get_connection()

    print("Database connected!")

    connection.close()