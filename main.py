
from fasthtml.common import *
from pymongo import MongoClient
from bson.objectid import ObjectId
import os

# MongoDB configuration 
MONGO_URI = os.environ.get('MONGO_URI')
if not MONGO_URI:
    raise ValueError("MONGODB_URI environment variable is not set")
database = 'test_db'
collection = 'users'

# Database setup
def setup_db(database, collection):
    try:
        client = MongoClient(MONGO_URI)
        # Create/get database
        db = client[database]
        # Explicitly create collection
        if collection not in db.list_collection_names():
            db.create_collection(collection)
        return db
    except Exception as e:
        print(f"Failed to setup database: {str(e)}")
        return None

db = setup_db(database, collection)

# Initialize FastHTML
app = FastHTML(
    hdrs=(picolink), 
    debug=True
)

@app.route("/")
def get():
    try:
        return Div(f"Your available collections are: {db.list_collection_names()}", style="color: green")
    except Exception as e:
        return Div(f"Database connection failed: {str(e)}", style="color: red")

serve()
