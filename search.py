
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
                          type="button")

    search_form = Grid(
        Div(search_input, cls="col-span-5"),
        Div(search_button, cls="col-span-1"),
        cols=6,
        hx_get="/search/results",
        hx_target="#search-results",
        hx_include="closest form",
        hx_trigger="submit, click from:.primary, keyup[key=='Enter'] from:input[name='q']",
        cls="items-center gap-2")

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
                    "k": 5,
                    "similarity": "cosine"
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

    # Perform hybrid search
    pipeline = [
        {
            "$search": {
                "index": "vector_index", # We'll use the vector index but with compound operators
                "compound": {
                    "should": [
                        {
                            "text": {
                                "query": query,
                                "path": "text",
                                "score": { "boost": { "value": 1.5 } }
                            }
                        },
                        {
                            "knnBeta": {
                                "vector": query_embedding,
                                "path": "embedding",
                                "k": 10,
                                "similarity": "cosine"
                            }
                        }
                    ]
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
                "score": { "$meta": "searchScore" }
            }
        }
    ]
    try:
        collection = mongodb_client[db_name]['embeddings']
        results = list(collection.aggregate(pipeline))
        return results
    except Exception as e:
        print(f"Hybrid search error for query '{query}': {str(e)}")
        return []
