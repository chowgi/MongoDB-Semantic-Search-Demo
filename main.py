from fasthtml.common import *
from monsterui.all import *
from pymongo import MongoClient

# MongoDB URI
MONGO_URI = os.environ.get('MONGO_URI')
if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable is not set")

# Connect to MongoDB
def setup_db():
    try:
        client = MongoClient(MONGO_URI)
        db = client['my_db_name']
        print(f"Connectioned to MongoDB: {MONGO_URI}")
        return db
    
    except Exception as e:
        print(f"Failed to initialize MongoDB connection: {e}")
        raise

db = setup_db()

# Initialize FastHTML
app = FastHTML(hdrs=Theme.blue.headers(),debug=True)

# Allow static files to be served
app.mount("/static", StaticFiles(directory="static"), name="static")

#App routes
@app.get("/")
def home():
    return "<h1>Hello, World</h1>"

serve()