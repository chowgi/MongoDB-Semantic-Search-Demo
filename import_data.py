
import json
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables for MongoDB connection 
load_dotenv()

# Connect to MongoDB
client = MongoClient(os.environ['WORK_CLUSTER'])
db = client.retail
collection = db.products

# Read JSON file
with open('sample.json', 'r', encoding='utf-8') as file:
    # Read each line and insert into MongoDB
    for line in file:
        if line.strip():  # Skip empty lines
            product = json.loads(line)
            collection.insert_one(product)

print("Data import completed successfully!")
