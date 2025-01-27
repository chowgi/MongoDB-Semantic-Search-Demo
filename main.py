
from pymongo import MongoClient
from bson import ObjectId
import os
from dotenv import load_dotenv

# Load environment variables for MongoDB connection
load_dotenv()

# You can use any database you want - here we're using MongoDB which is great for document storage
# MongoDB automatically creates databases and collections when they're first used
client = MongoClient(os.environ['MONGO_URI'])
db = client.amazon
collection = products
