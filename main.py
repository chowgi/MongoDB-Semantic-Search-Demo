
from fasthtml.common import *
from pymongo import MongoClient
from bson.objectid import ObjectId
from pymongo.synchronous import collection
import os

# Constants
MONGO_URI = os.environ.get('MONGO_URI')
if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable is not set")
dbname = 'todo'

# Database setup
def setup_db(db_name):
    try:
        client = MongoClient(MONGO_URI)
        db = client[db_name]
        return client, db.recipes
    except Exception as e:
        print(f"Failed to initialize MongoDB connection: {e}")
        raise

dbname = 'recipes'
mongo_client, recipes = setup_db()

# Initialize FastHTML
app = FastHTML(
    hdrs=(picolink), 
    debug=True
)



serve()