
from fasthtml.common import *
from pymongo import MongoClient
from bson.objectid import ObjectId
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
MONGO_URI = os.environ.get('MONGODB_URI')
if not MONGO_URI:
    raise ValueError("MONGODB_URI environment variable is not set")

# Database setup
def setup_db():
    try:
        client = MongoClient(MONGO_URI)
        db = client.todos_db
        # Test connection
        client.server_info()
        return client, db
    except Exception as e:
        print(f"Failed to initialize MongoDB connection: {e}")
        raise

client, db = setup_db()

# Initialize FastHTML
app = FastHTML(
    hdrs=(picolink), 
    debug=True
)

@app.route("/health")
def get():
    try:
        client.server_info()
        return Div("Database connection successful!", style="color: green")
    except Exception as e:
        return Div(f"Database connection failed: {str(e)}", style="color: red")

serve()
