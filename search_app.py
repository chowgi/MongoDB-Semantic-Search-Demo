
from fasthtml.common import *
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.llms.openai import OpenAI
import os
import pymongo

# Initialize FastHTML with MonsterUI theme
hdrs = Theme.green.headers()
app, rt = fast_app(hdrs=hdrs, static_path="public", live=True, debug=True, title="MongoDB + Voyage AI Search Demo")

# Retrieve environment variables for necessary API keys and URIs
openai_api_key = os.environ['OPENAI_API_KEY']
mongodb_uri = os.environ['MONGODB_URI']
voyage_api_key = os.environ['VOYAGE_API_KEY']

def init_search():
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
        db_name="hawthornfc", 
        collection_name='embeddings',
        embedding_key="embedding",
        text_key="text",
        fulltext_index_name="text_index",
        metadata_key="metadata",
        insert_metadata_if_missing=True
    )

    # Initialize the storage context for vector store operations
    storage_context = StorageContext.from_defaults(vector_store=store)

    # Generate the vector index from the existing vector store
    index = VectorStoreIndex.from_vector_store(store)

    return {
        "mongodb_client": mongodb_client,
        "store": store,
        "storage_context": storage_context,
        "index": index,
        "db_name": "hawthornfc"
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

def navbar():
    return NavBar(A("Search",href='/'),
                  brand=A(
                      DivLAligned(
                      Img(src='MongoDB.svg', height=25,width=25),
                      H4('MongoDB + Voyage AI Search')
                    ), href="/"
                 )
            )

def search_bar():
    search_input = Input(type="search",
                         name="query",
                         placeholder="Search documents...",
                         cls="search-bar",
                         id="search-input")
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
        hx_trigger="submit, keyup[key=='Enter'] from:input[name='query']",
        hx_indicator="#loading"
    )

    return Div(search_form, cls='pt-5')

# Initialize search components
search_components = init_search()
index = search_components["index"]

@rt('/')
def get():
    """Main search page that displays the search form and empty results container"""
    search_results = Div(id="search-results", cls="m-2")

    return Title("Search - MongoDB + Voyage AI"), Container(
        navbar(),
        Div(H2("MongoDB Atlas Search Comparison", cls="pb-10 text-center"),
            P("Compare Text, Vector, and Hybrid Search Methods", cls="pb-5 text-center uk-text-lead"),
            search_bar(),
            cls="container mx-auto p-4"), # Added container for styling
        Div(id="search-results", cls="m-2"),
        Div(Loading(cls=LoadingT.dots), id="loading", cls="htmx-indicator"),
        cls=ContainerT.lg
    )

@rt("/search/results")
def get(query: str = None, request=None):
    
    clear_search_bar = Input(type="search",
         name="query",
         placeholder="Search documents...",
         cls="search-bar",
         id="search-input",
         hx_swap_oob="true")
    
    if query:
        # Use the search function
        results = search(query, index, top_k=3)

        # Create a card for each mode with the mode_name as the title
        cards = []  # Initialize the cards list
        for mode_name, nodes in results.items(): 

            card_title = H4(f"Mode: {mode_name}")
            card_content = []
            for node in nodes:

                node_content = Div(
                    P(Strong("Retrieved Node:"), f" {node.node.text[:200]}", cls=TextT.sm),
                    P(Strong("Score:"), f" {node.score}", cls=TextT.sm),
                    P(Strong("Source:"), " ", 
                      A(
                        node.metadata['url'],
                        href=node.metadata['url'],
                        target='_blank',
                        cls="text-primary"
                      ),
                      cls=TextT.sm
                    )
                )
                card_content.append(node_content)

            # Add the completed card with a title and content to the list
            cards.append(Card(card_title, *card_content))

        grid = Grid(*cards, cols_lg=3, cls="gap-4")  # Display in a 3-column grid

        return clear_search_bar, grid
    else:
        return P("Please enter a search query.")

# Start the server on port 3030
if __name__ == "__main__":
    serve(app=app, host="0.0.0.0", port=3030)
