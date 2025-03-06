from fasthtml.common import *
from monsterui.all import *
import os
import pymongo
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.llms.openai import OpenAI
from trafilatura import fetch_url, extract
from trafilatura.sitemaps import sitemap_search

# utils.py content
def initialize_settings(openai_api_key, voyage_api_key):
    Settings.llm = OpenAI(
        temperature=0.7, model="gpt-3.5-turbo", api_key=openai_api_key
    )
    Settings.embed_model = VoyageEmbedding(
        voyage_api_key=voyage_api_key,
        model_name="voyage-3",
    )

def setup_vector_search(mongodb_uri, db_name, website_url):
    mongodb_client = pymongo.MongoClient(mongodb_uri)
    store = MongoDBAtlasVectorSearch(mongodb_client, db_name=db_name, collection_name='embeddings')
    storage_context = StorageContext.from_defaults(vector_store=store)
    index = VectorStoreIndex.from_vector_store(store)
    chat_engine = index.as_query_engine(similarity_top_k=3)
    return mongodb_client, store, index, chat_engine


# ui.py content
def create_message_div(role, content):
    return Div(
        Div(role, cls="chat-header"),
        Div(content, cls=f"chat-bubble chat-bubble-{'primary' if role == 'user' else 'secondary'}"),
        cls=f"chat chat-{'end' if role == 'user' else 'start'}")

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

    return Grid(*product_cards, cols_lg=3)

def create_chat_interface():
    return Card(
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
    )

class Loading(Div):
    def __init__(self):
        super().__init__("Loading...", cls=("text-center", "text-lg"))

# Initialize FastHTML with MonsterUI theme
hdrs = Theme.green.headers()
app, rt = fast_app(hdrs=hdrs, static_path="public", live=True, debug=True)

# Retrieve environment variables for necessary API keys and URIs
openai_api_key = os.environ['OPENAI_API_KEY']
mongodb_uri = os.environ['MONGODB_URI']
voyage_api_key = os.environ['VOYAGE_API_KEY']
website_url = "https://www.hawthornfc.com.au/sitemap/index.xml"
db_name = "bendigo"

# Configure the default LLM and embedding model settings
initialize_settings(openai_api_key, voyage_api_key)

# Set up MongoDB Atlas Vector Search and initialize components
mongodb_client, store, index, chat_engine = setup_vector_search(mongodb_uri, db_name, website_url)

##################################################
################  Route Handlers   ###############
##################################################

@rt("/")
def get():
    return Container(
        navbar(),
        use_case_cards(),
        cls=ContainerT.sm
    )

@rt("/search")
def get():
    return Container(
        navbar(),
        P("Search demo coming soon! Maybe..."),
        cls=ContainerT.sm
    )

@rt("/rag")
def get():
    return Container(
        navbar(),
        create_chat_interface(),
        cls=ContainerT.sm
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
async def post(message: str):
    try:
        ai_response = await chat_engine.aquery(message)

        # Create a response that includes the source URLs
        source_content = ""
        if hasattr(ai_response, 'source_nodes') and ai_response.source_nodes:
            source_content = "<br><br><strong>Sources:</strong><br>"
            for i, source in enumerate(ai_response.source_nodes):
                if 'url' in source.metadata:
                    source_content += f"{i+1}. <a href='{source.metadata['url']}' target='_blank'>{source.metadata['url']}</a><br>"

        # Create the full response with main content and sources
        full_response = f"{ai_response.response}{source_content}"

        response_div = await create_message_div("assistant", full_response)
        return (
            response_div,
            Div(id="loading", hx_swap_oob="true")
        )
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        error_div = await create_message_div("assistant", f"Sorry, I encountered an error: {str(e)}")
        return (
            error_div,
            Div(id="loading", hx_swap_oob="true")
        )

serve()