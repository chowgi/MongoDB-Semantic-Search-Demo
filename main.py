
from pymongo import MongoClient
from fasthtml.common import *
import os

from pymongo.synchronous import collection

# MongoDB URI
MONGO_URI = os.environ.get('MONGO_URI')
if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable is not set")

# Connect to MongoDB
def setup_db():
    try:
        client = MongoClient(MONGO_URI)
        db = client['my_db_name']
        print(f"Connctioned to MongoDB: {MONGO_URI}")
        return db
    
    except Exception as e:
        print(f"Failed to initialize MongoDB connection: {e}")
        raise

setup_db()
