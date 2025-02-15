
from fasthtml.common import *
from monsterui.all import *
from pymongo import MongoClient
from datetime import datetime

# MongoDB URI
MONGO_URI = os.environ.get('MONGO_URI')
if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable is not set")

# Connect to MongoDB
def setup_db():
    try:
        client = MongoClient(MONGO_URI)
        db = client['my_db_name']
        print(f"Connected to MongoDB: {MONGO_URI}")
        return db
    
    except Exception as e:
        print(f"Failed to initialize MongoDB connection: {e}")
        raise

db = setup_db()

# Initialize FastHTML with MonsterUI theme
app, rt = fast_app(hdrs=Theme.blue.headers(daisy=True), static_path="static", debug=True)

#App routes
@rt("/")
def home():
    return Titled("Your First App",
      Card(
          H1("Welcome!"),
          P("Your first MonsterUI app", cls=TextPresets.muted_sm),
          P("I'm excited to see what you build with MonsterUI!"),
          Img(src="/static/dog.jpg", alt="Dog picture", cls="w-full rounded-lg")))
          

serve()
