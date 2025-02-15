
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
app = FastHTML(hdrs=Theme.blue.headers(daisy=True), debug=True)

# Allow static files to be served
app.mount("/static", StaticFiles(directory="static"), name="static")

#App routes
@app.get("/")
def home():
    return Container(
        # Header Section
        Section(
            H1("Welcome to FastHTML MongoDB", cls=TextT.center),
            P("A powerful web application template with MongoDB integration", 
              cls=(TextT.center, TextT.muted, TextT.lg)),
            cls=SectionT.primary
        ),
        
        # Features Grid
        Grid(
            Card(
                CardHeader(H3("MongoDB Ready")),
                CardBody(
                    P("Pre-configured MongoDB integration with error handling and environment variable support"),
                    Button("Learn More", cls=ButtonT.primary)
                ),
                cls=CardT.hover
            ),
            Card(
                CardHeader(H3("MonsterUI Styling")),
                CardBody(
                    P("Beautiful components and themes powered by MonsterUI"),
                    Button("Explore", cls=ButtonT.secondary)
                ),
                cls=CardT.hover
            ),
            Card(
                CardHeader(H3("Fast Development")),
                CardBody(
                    P("Static file serving, debug mode, and more features to speed up your development"),
                    Button("Get Started", cls=ButtonT.primary)
                ),
                cls=CardT.hover
            ),
            cols=3
        ),
        
        # Stats Section
        Section(
            Grid(
                Card(H3("Easy Setup"), P("Configure in minutes"), cls=CardT.primary),
                Card(H3("Production Ready"), P("Deploy with confidence"), cls=CardT.secondary),
                Card(H3("Extensible"), P("Build anything"), cls=CardT.hover),
                cols=3
            ),
            cls=SectionT.muted
        ),
        
        # Footer
        Footer(
            DivCentered(
                P(f"© {datetime.now().year} FastHTML MongoDB Template. Made with ❤️", 
                  cls=TextT.center)
            ),
            cls="mt-8 py-4"
        )
    )

serve()
