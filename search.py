
from llama_index.core import Settings
from llama_index.embeddings.voyageai import VoyageEmbedding
from fasthtml.common import *
from monsterui.all import *

def search_bar():
    search_input = Input(type="search",
                         name="q",
                         placeholder="Search documents...",
                         cls="search-bar")
    search_button = Button("Search", 
                          cls=ButtonT.primary,
                          type="submit")

    search_form = Form(
        Grid(
            Div(search_input, cls="col-span-5"),
            Div(search_button, cls="col-span-1"),
            cols=6,
            cls="items-center gap-2"
        ),
        hx_get="/search/results",
        hx_target="#search-results",
        hx_trigger="submit, keyup[key=='Enter'] from:input[name='q']"
    )

    return Div(search_form, cls='pt-5')

def text_search(query: str, mongodb_client, db_name):
    """Search using MongoDB Atlas Text Search with text index"""
    print("text search ran")
    pipeline = [
        {
            "$search": {
                "index": "text_index",
                "text": {
                    "query": query,
                    "path": "text"
                },
                "highlight": {
                    "path": "text"
                }
            }
        },
        {
            "$limit": 5
        },
        {
            "$project": {
                "text": 1,
                "url": 1,
                "score": { "$meta": "searchScore" },
                "highlights": { "$meta": "searchHighlights" }
            }
        }
    ]
    try:
        collection = mongodb_client[db_name]['embeddings']
        results = list(collection.aggregate(pipeline))
        return results
    except Exception as e:
        print(f"Text search error for query '{query}': {str(e)}")
        return []

def vector_search(query: str, mongodb_client, db_name):
    """Search using MongoDB Atlas Vector Search with vector index"""
    print("vector search ran")
    # First, generate embedding for the query
    query_embedding = Settings.embed_model.get_text_embedding(query)

    # Perform vector search
    pipeline = [
        {
            "$search": {
                "index": "vector_index",
                "knnBeta": {
                    "vector": query_embedding,
                    "path": "embedding",
                    "k": 5
                }
            }
        },
        {
            "$project": {
                "text": 1,
                "url": 1,
                "score": { "$meta": "searchScore" }
            }
        }
    ]
    try:
        collection = mongodb_client[db_name]['embeddings']
        results = list(collection.aggregate(pipeline))
        return results
    except Exception as e:
        print(f"Vector search error for query '{query}': {str(e)}")
        return []

def hybrid_search(query: str, mongodb_client, db_name):
    """Hybrid search using both text and vector search capabilities"""
    print("hybrid search ran")
    # First, generate embedding for the query
    query_embedding = Settings.embed_model.get_text_embedding(query)

    # Perform hybrid search - Use a different approach without nesting knnBeta
    # MongoDB Atlas doesn't allow knnBeta in compound queries
    pipeline = [
        {
            "$search": {
                "index": "vector_index",
                "text": {
                    "query": query,
                    "path": "text"
                }
            }
        },
        {
            "$limit": 10
        },
        {
            "$project": {
                "text": 1,
                "url": 1,
                "score": { "$meta": "searchScore" }
            }
        }
    ]
    
    # Execute a separate vector search to combine results later
    vector_pipeline = [
        {
            "$search": {
                "index": "vector_index",
                "knnBeta": {
                    "vector": query_embedding,
                    "path": "embedding",
                    "k": 10
                }
            }
        },
        {
            "$project": {
                "text": 1,
                "url": 1,
                "score": { "$meta": "searchScore" }
            }
        }
    ]
    
    try:
        collection = mongodb_client[db_name]['embeddings']
        text_results = list(collection.aggregate(pipeline))
        vector_results = list(collection.aggregate(vector_pipeline))
        
        # Combine and deduplicate results
        seen_urls = set()
        combined_results = []
        
        # Add text results first (giving them priority)
        for result in text_results:
            if result.get('url') not in seen_urls:
                seen_urls.add(result.get('url'))
                combined_results.append(result)
        
        # Add vector results next
        for result in vector_results:
            if result.get('url') not in seen_urls:
                seen_urls.add(result.get('url'))
                combined_results.append(result)
        
        # Return only up to 5 results
        return combined_results[:5]
    except Exception as e:
        print(f"Hybrid search error for query '{query}': {str(e)}")
        return []
