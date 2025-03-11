
from fasthtml.common import *
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.llms.openai import OpenAI
import os
import pymongo

# Initialize FastHTML with MonsterUI theme
hdrs = Theme.green.headers()
app, rt = fast_app(hdrs=hdrs, static_path="public", live=True, debug=True, title="MongoDB + Voyage AI RAG Demo")

# Retrieve environment variables for necessary API keys and URIs
openai_api_key = os.environ['OPENAI_API_KEY']
mongodb_uri = os.environ['MONGODB_URI']
voyage_api_key = os.environ['VOYAGE_API_KEY']

def init_rag():
    """Initialize RAG components and return the configured chat engine"""
    
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

    # Create the chat engine
    chat_engine = index.as_query_engine(similarity_top_k=3)
    
    return {
        "chat_engine": chat_engine
    }

def navbar():
    return NavBar(A("RAG",href='/'),
                  brand=A(
                      DivLAligned(
                      Img(src='MongoDB.svg', height=25,width=25),
                      H4('MongoDB + Voyage AI RAG')
                    ), href="/"
                 )
            )

def create_message_div(role, content):
    """Create a chat message div with proper styling"""
    return Div(
        Div(role, cls="chat-header"),
        Div(content, cls=f"chat-bubble chat-bubble-{'primary' if role == 'user' else 'secondary'}"),
        cls=f"chat chat-{'end' if role == 'user' else 'start'}")

# Initialize RAG components
rag_components = init_rag()
chat_engine = rag_components["chat_engine"]

@rt('/')
def get():
    return Container(
        navbar(),
        Div(H2("Resource Augmented Generation", cls="pb-10 text-center"),
            P("Deliver contextually relevant and accurate responses based on up-to-date private data source.", cls="pb-5 text-center uk-text-lead")),
        Card(
            Div(id="chat-messages", 
                cls="space-y-4 h-[60vh] overflow-y-auto p-4",
                style="overflow: auto"
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

@rt("/send-message")
def post(message: str):
    return (
        create_message_div("user", message),
        TextArea(id="message", placeholder="Type your message...", hx_swap_oob="true"),
        Div(hx_trigger="load", hx_post="/get-response", hx_vals=f'{{"message": "{message}"}}',
            hx_target="#chat-messages", hx_swap="beforeend scroll:#chat-messages:bottom")
    ),Div(Loading(cls=LoadingT.dots), id="loading")

@rt("/get-response")
def post(message: str):
    # Process the chat message
    ai_response = chat_engine.query(message)

    return (
        create_message_div("assistant", ai_response),
        Div(id="loading", hx_swap_oob="true"))

# Start the server on port 3000
if __name__ == "__main__":
    serve(app=app, host="0.0.0.0", port=3000)
