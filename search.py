
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import StorageContext, Settings
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.llms.openai import OpenAI
import pymongo

def init_search(mongodb_uri, voyage_api_key, openai_api_key, collection_name="embeddings", db_name="mongo_voyage_demos"):
    """
    Initialize search components for MongoDB Atlas Vector Search with Voyage AI embeddings.
    
    Args:
        mongodb_uri: URI for MongoDB connection
        voyage_api_key: API key for VoyageAI embeddings
        openai_api_key: API key for OpenAI
        collection_name: Name of the MongoDB collection to use
        db_name: Name of the MongoDB database
        
    Returns:
        Dict containing initialized components for search functionality
    """
    # Configure the default Language Model with OpenAI's API
    Settings.llm = OpenAI(
        temperature=0.7, model="gpt-3.5-turbo", api_key=openai_api_key
    )

    # Set the default embedding model using VoyageAI Embedding
    Settings.embed_model = VoyageEmbedding(
        voyage_api_key=voyage_api_key,
        model_name="voyage-3",
    )

    # Establish MongoDB client connection using the provided URI
    mongodb_client = pymongo.MongoClient(mongodb_uri)

    # Set up MongoDB Atlas Vector Search connection with specified database and collection
    store = MongoDBAtlasVectorSearch(
        mongodb_client, 
        db_name=db_name, 
        collection_name=collection_name,
        embedding_key="embedding",
        text_key="text",
        fulltext_index_name="text_index",
        metadata_key="metadata",
        insert_metadata_if_missing=True
    )

    # Initialize the storage context for vector store operations
    storage_context = StorageContext.from_defaults(vector_store=store)
    
    # Return the components needed for search functionality
    return {
        "mongodb_client": mongodb_client,
        "store": store,
        "storage_context": storage_context,
    }
