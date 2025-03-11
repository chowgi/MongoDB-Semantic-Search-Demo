
import os
import pymongo
from llama_index.core import Settings
from llama_index.embeddings.voyageai import VoyageEmbedding
from typing import List, Dict, Any

def create_vector_embeddings(
    mongodb_uri: str,
    db_name: str,
    collection_name: str,
    text_fields: List[str],
    voyage_api_key: str
):
    """
    Creates vector embeddings for text fields in a MongoDB collection.
    
    Args:
        mongodb_uri: MongoDB connection URI
        db_name: Database name
        collection_name: Collection name to process
        text_fields: List of field names containing text to embed
        voyage_api_key: API key for VoyageAI
    """
    # Configure the embedding model
    Settings.embed_model = VoyageEmbedding(
        voyage_api_key=voyage_api_key,
        model_name="voyage-3",
    )
    
    # Establish MongoDB client connection
    mongodb_client = pymongo.MongoClient(mongodb_uri)
    db = mongodb_client[db_name]
    collection = db[collection_name]
    
    # Create vector search index if it doesn't exist
    try:
        # Check if index already exists
        existing_indexes = collection.list_indexes()
        index_exists = any(idx.get('name') == 'vector_index' for idx in existing_indexes)
        
        if not index_exists:
            collection.create_index(
                [("embedding", pymongo.TEXT)],
                name="text_index"
            )
            
            # Create vector search index
            db.command({
                "createIndexes": collection_name,
                "indexes": [{
                    "name": "vector_index",
                    "key": {"embedding": "vector"},
                    "vectorOptions": {
                        "type": "cosine",
                        "dimensions": 1024
                    }
                }]
            })
            print(f"Vector search index created for collection '{collection_name}'")
        else:
            print(f"Vector index already exists for collection '{collection_name}'")
    
    except Exception as e:
        print(f"Error creating vector index: {str(e)}")
    
    # Get documents that don't have embeddings yet
    docs_to_embed = collection.find({"embedding": {"$exists": False}})
    doc_count = collection.count_documents({"embedding": {"$exists": False}})
    
    print(f"Found {doc_count} documents without embeddings")
    
    # Process documents in batches
    batch_size = 20
    processed = 0
    
    for doc in docs_to_embed:
        # Combine all text fields into one string
        text_content = " ".join([str(doc.get(field, "")) for field in text_fields if doc.get(field)])
        
        if not text_content.strip():
            print(f"No text content found in document {doc['_id']}, skipping")
            continue
        
        # Generate embedding
        try:
            embedding = Settings.embed_model.get_text_embedding(text_content)
            
            # Update document with embedding
            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"embedding": embedding}}
            )
            
            processed += 1
            if processed % 10 == 0:
                print(f"Processed {processed} documents")
                
        except Exception as e:
            print(f"Error embedding document {doc['_id']}: {str(e)}")
    
    print(f"Completed processing. Added embeddings to {processed} documents.")

if __name__ == "__main__":
    # Get required environment variables or set defaults
    mongodb_uri = os.environ.get('MONGODB_URI')
    voyage_api_key = os.environ.get('VOYAGE_API_KEY')
    
    if not mongodb_uri or not voyage_api_key:
        print("Error: MONGODB_URI and VOYAGE_API_KEY environment variables must be set")
        exit(1)
    
    # Example usage
    db_name = input("Enter database name: ")
    collection_name = input("Enter collection name: ")
    
    # Get text fields to embed
    text_fields = []
    print("Enter text field names (one per line, blank line to finish):")
    while True:
        field = input()
        if not field:
            break
        text_fields.append(field)
    
    if not text_fields:
        print("Error: At least one text field must be specified")
        exit(1)
    
    print(f"Creating embeddings for fields {text_fields} in collection {collection_name}")
    create_vector_embeddings(mongodb_uri, db_name, collection_name, text_fields, voyage_api_key)
