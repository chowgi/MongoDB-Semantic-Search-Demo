import json
import os
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from pymongo.errors import BulkWriteError
from bson import json_util, ObjectId

def load_data():
    # Get MongoDB URI from environment variable
    mongodb_uri = os.environ.get('MONGODB_URI')
    if not mongodb_uri:
        raise ValueError("Please set the MONGODB_URI environment variable")

    # Connect to MongoDB
    client = MongoClient(mongodb_uri)
    db = client['semantic_search_demo']
    collection = db['movie_embeddings']

    # Read JSON data using json_util to properly handle MongoDB extended JSON
    with open('movie_embeddings.json', 'r') as f:
        documents = json_util.loads(f.read())

    # Remove _id field to let MongoDB auto-generate it and check for existing documents
    new_documents = []
    for document in documents:
        # Check if document already exists
        if not collection.find_one({"metadata.title": document.get("metadata", {}).get("title")}):
            if '_id' in document:
                del document['_id']
            new_documents.append(document)
        else:
            print(f"Skipping {document.get('metadata', {}).get('title')} - Document already exists")

    if new_documents:
        try:
            # Insert only new documents
            result = collection.insert_many(new_documents)
            print(f"Successfully inserted {len(result.inserted_ids)} documents")
        except BulkWriteError as e:
            print(f"Error inserting documents: {e.details}")
        finally:
            client.close()
    else:
        print("No new documents to insert")

def create_index():
    # Get MongoDB URI from environment variable
    mongodb_uri = os.environ.get('MONGODB_URI')
    if not mongodb_uri:
        raise ValueError("Please set the MONGODB_URI environment variable")

    # Connect to MongoDB
    client = MongoClient(mongodb_uri)
    db = client['semantic_search_demo']
    collection = db['movie_embeddings']

    # Check and create text index if it doesn't exist
    try:
        existing_indexes = collection.list_search_indexes()
        text_index_exists = any(index['name'] == 'text_index' for index in existing_indexes)

        if not text_index_exists:
            text_search_model = SearchIndexModel(
                definition={
                    "mappings": {
                        "dynamic": False,
                        "fields": {
                            "text": {
                                "type": "string"
                            }
                        }
                    }
                },
                name="text_index"
            )
            collection.create_search_index(text_search_model)
            print("Created text search index")
        else:
            print("Text search index already exists")
    except Exception as e:
        print(f"Error managing text search index: {str(e)}")

    # Create vector search index if it doesn't exist
    try:
        # List all search indexes
        existing_indexes = collection.list_search_indexes()
        vector_index_exists = any(index['name'] == 'vector_index' for index in existing_indexes)

        if not vector_index_exists:
            search_model = SearchIndexModel(
                definition={
                  "fields": [
                    {
                      "numDimensions": 1024,
                      "path": "embedding",
                      "similarity": "cosine",
                      "type": "vector"
                    },
                    {
                      "path": "metadata.rating",
                      "type": "filter"
                    },
                    {
                      "path": "metadata.languages",
                      "type": "filter"
                    }
                  ]
                },
                name="vector_index",
                type="vectorSearch"
            )
            collection.create_search_index(search_model)
            print("Created vector search index")
        else:
            print("Vector search index already exists")
    except Exception as e:
        print(f"Error managing search index: {str(e)}")

if __name__ == "__main__":
    load_data()
    create_index()
