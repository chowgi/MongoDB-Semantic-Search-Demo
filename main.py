from fasthtml.common import *
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.llms.openai import OpenAI
# from llama_index.core import Document
from llama_index.core import VectorStoreIndex, StorageContext
#from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI
from monsterui.all import *
import pymongo
import os

# Initialize FastHTML with MonsterUI theme
hdrs = Theme.green.headers()
app, rt = fast_app(hdrs=hdrs, static_path="public", live=True, debug=True)

# Retrieve environment variables for necessary API keys and URIs
openai_api_key = os.environ['OPENAI_API_KEY']
mongodb_uri = os.environ['MONGODB_URI']
voyage_api_key = os.environ['VOYAGE_API_KEY']
website_url = "https://www.hawthornfc.com.au/sitemap/index.xml"
db_name = "bendigo"

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
store = MongoDBAtlasVectorSearch(mongodb_client, db_name=db_name, collection_name='embeddings')

# Initialize the storage context for vector store operations
storage_context = StorageContext.from_defaults(vector_store=store)

# Generate the vector index from the existing vector store
index = VectorStoreIndex.from_vector_store(store)

# create the chat engine
chat_engine = index.as_query_engine(similarity_top_k=3)

##################################################
########## Settings and Admin Logic ##############
##################################################



##################################################
################# Search Logic ###################
##################################################

def search_bar():
    search_input = Input(type="search",
                         name="query",
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
        hx_trigger="submit, keyup[key=='Enter'] from:input[name='query']"
    )

    return Div(search_form, cls='pt-5')

def search(query, top_k=5):
    """
    Retrieve nodes using different query modes and return results in a dictionary.

    Args:
        query (str): The query string
        top_k (int): Number of top results to retrieve

    Returns:
        dict: Dictionary with keys as modes and values as retrieved nodes
    """
    modes = ["text_search", "default", "hybrid"]  # default is vector
    results = {}

    for mode in modes:
        # Create a retriever with the specific mode
        retriever = index.as_retriever(
            similarity_top_k=top_k,
            vector_store_query_mode=mode
        )

        # Retrieve nodes using the current mode
        retrieved_nodes = retriever.retrieve(query)

        # Map 'default' to 'vector' in the results dictionary for clarity
        mode_name = "vector" if mode == "default" else mode

        # Store the retrieved nodes in the results dictionary
        results[mode_name] = retrieved_nodes

    return results

##################################################
################## RAG Logic #####################
##################################################

def create_message_div(role, content):
    return Div(
        Div(role, cls="chat-header"),
        Div(content, cls=f"chat-bubble chat-bubble-{'primary' if role == 'user' else 'secondary'}"),
        cls=f"chat chat-{'end' if role == 'user' else 'start'}")

##################################################
################## Agent Logic ###################
##################################################




##################################################
##############  Nav And Home Page ################
##################################################

def navbar():
    return NavBar(A("Search",href='/search'),
                  A("RAG",href='/rag'),
                  A("Agents",href='/agents'),
                  A("Settings",href='/settings'),
                  brand=A(
                      DivLAligned(
                      Img(src='MongoDB.svg', height=25,width=25),
                      H4('MongoDB + Voyage AI')
                    ), href="/"
                 )
            )

def use_case_cards():
    products = [
        {"name": "Semantic Search", "description": "MongoDB Atlas Vector Search enables semantic similarity searches on data by converting text, images, or other media into vector embeddings, allowing queries based on meaning rather than exact matches", "img": "/vector_icon.png", "url": "/search"},
        {"name": "Retrieval Augmented Generation (RAG)","description": "A MongoDB-powered RAG demo showcases seamless integration of vector search and document retrieval to enhance AI-generated responses with real-time, context-aware information", "img": "/rag_icon.png", "url": "/rag"},
        {"name": "Agentic AI","description": "The demo integrates MongoDB Atlas with AutoGen to create a powerful multi-agent system that combines flexible agent orchestration with robust data management and vector search capabilities", "img": "/ai_icon.png", "url": "/agents"}
    ]

    product_cards = [
        Card(
            Img(src=p["img"], alt=p["name"], style="height:100px; object-fit:cover; display:block; margin:auto;"),
            H4(p["name"], cls="mt-2"),
            P(p["description"]),
            Button("See Demo", cls=(ButtonT.primary, "mt-2"), onclick=f"window.location.href='{p['url']}'")
        ) for p in products
    ]

    return Grid(*product_cards, cols_lg=3,cls='pt-20 gap-4')

##################################################
###################  Routes ######################
##################################################

@rt("/")
def get():
    return Container(
        navbar(),
        use_case_cards(),
        cls=ContainerT.lg
    )

@rt("/search")
def get():
    """Main search page that displays the search form and empty results container"""
    search_results = Div(id="search-results", cls="m-2")

    return Container(
        navbar(),
        Div(H2("MongoDB Atlas Search Comparison", cls="pb-10"),
            P("Compare Text, Vector, and Hybrid Search Methods", cls="pb-5"),
            search_bar(),
            cls="container mx-auto p-4"), # Added container for styling
        Div(id="search-results", cls="m-2"),
        cls=ContainerT.lg
    )


@rt("/search/results")
def get(query: str = None, request=None):
    if query:
        results = search(query, top_k=3) #Reduced top_k for grid display

        # Create a dictionary to track seen content to avoid duplicates
        # Use text content as the key since that's what appears to be duplicated
        seen_texts = {}
        cards = []
        
        for mode, nodes in results.items():
            for node in nodes:
                # Skip this node if we've already seen this text
                if node.text in seen_texts:
                    continue
                
                # Mark this text as seen
                seen_texts[node.text] = True
                
                card_content = Div(
                    H4(f"Mode: {mode}"),
                    P(node.text),
                    P(f"Score: {node.score}" if hasattr(node, 'score') else "")
                )
                cards.append(Card(card_content))

        grid = Grid(*cards, cols_lg=3, cls="gap-4") # Display in a 3-column grid

        return grid
    else:
        return P("Please enter a search query.")


@rt("/rag")
def get():
    return Container(
        navbar(),
        Card(
            Div(id="chat-messages", 
                cls="space-y-4 h-[60vh] overflow-y-auto p-4",
                style="height:300px; overflow: auto"
               ),
            Form(
                TextArea(id="message", placeholder="Type your message..."),
                Button(
                    "Send",
                    cls=ButtonT.primary,
                    hx_post="/send-message",
                    hx_target="#chat-messages",
                    hx_swap="beforeend scroll:#chat-messages:bottom"
                ),
                cls="space-y-2",
                hx_trigger="keydown[key=='Enter' && !shiftKey]",
                hx_post="/send-message",
                hx_target="#chat-messages",
                hx_swap="beforeend scroll:#chat-messages:bottom"
            )
        ),cls=ContainerT.lg
    )

@rt("/agents")
def get():
    return Container(
        navbar(),
        P("Agent demo coming soon! Maybe..."),
        cls=ContainerT.sm
    )

@rt("/settings")
def get():
    return Container(
        navbar(),
        P("Settings coming soon! Maybe..."),
        cls=ContainerT.sm
    )

@rt("/send-message")
def post(message: str):
    return (
        create_message_div("user", message),
        TextArea(id="message", placeholder="Type your message...", hx_swap_oob="true"),
        Div(hx_trigger="load", hx_post="/get-response", hx_vals=f'{{"message": "{message}"}}',
            hx_target="#chat-messages", hx_swap="beforeend scroll:#chat-messages:bottom")
    ),Div(Loading(), id="loading")

@rt("/get-response")
def post(message: str):

    ai_response = chat_engine.query(message)

    return (
        create_message_div("assistant", ai_response),
        Div(id="loading", hx_swap_oob="true"))


serve()