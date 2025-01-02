
from fasthtml.common import *
from pymongo import MongoClient
from bson.objectid import ObjectId
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
MONGO_URI = os.environ.get('MONGO_URI')
if not MONGO_URI:
    raise ValueError("MONGODB_URI environment variable is not set")
database = 'todo_db'
collection = 'todos'

# Database setup
def setup_db(database, collection):
        


db = setup_db(database, collection)


# Initialize FastHTML
app = FastHTML(
    hdrs=(picolink), 
    debug=True
)

@app.route("/")
def get():
    try:
        return Div(f"{client.list_database_names()}", style="color: green")
    except Exception as e:
        return Div(f"Database connection failed: {str(e)}", style="color: red")

serve()
