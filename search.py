
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.llms.openai import OpenAI
import os
import pymongo

# Initialize environment variables and connections
def init_search(mongodb_uri, voyage_api_key, openai_api_key, db_name="hawthornfc"):
    """Initialize search components and return the configured index"""
    
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
        collection_name='embeddings',
        embedding_key="embedding",
        text_key="text",
        fulltext_index_name="text_index",
    )

    # Initialize the storage context for vector store operations
    storage_context = StorageContext.from_defaults(vector_store=store)

    # Generate the vector index from the existing vector store
    index = VectorStoreIndex.from_vector_store(store)

    # Create the chat engine
    chat_engine = index.as_query_engine(similarity_top_k=3)

    return {
        "mongodb_client": mongodb_client,
        "store": store,
        "storage_context": storage_context,
        "index": index,
        "chat_engine": chat_engine,
        "db_name": db_name
    }

def search(query, index, top_k=5):
    """
    Perform search across multiple modes (text, vector, hybrid).
    
    Args:
        query: The search query
        index: VectorStoreIndex to use for search
        top_k: Maximum number of results to return per mode
        
    Returns:
        Dictionary of search results by mode
    """
    modes = ["text_search", "default", "hybrid"]  # default is vector
    results = {}  # Initialize results as an empty dictionary

    for mode in modes:
        try:
            # Create a retriever with the specific mode
            retriever = index.as_retriever(
                similarity_top_k=top_k,
                vector_store_query_mode=mode
            )

            # Retrieve nodes using the current mode
            retrieved_nodes = retriever.retrieve(query)

            # Map modes to titles for clarity
            mode_name = "Text Search" if mode == "text_search" else ("Vector Search" if mode == "default" else "Hybrid Search")

            # Store the retrieved nodes in the results dictionary using mode_name as key
            results[mode_name] = retrieved_nodes
        except Exception as e:
            # Log the error but continue with other modes
            print(f"Error with {mode} mode: {str(e)}")
            results[f"{mode} (Error)"] = []

    return results
