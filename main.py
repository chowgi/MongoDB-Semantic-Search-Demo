from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Document
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI
from trafilatura import fetch_url, extract
from trafilatura.sitemaps import sitemap_search
from fasthtml.common import *
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

# Delete a database
def delete_db(db_name: str):
    mongodb_client.drop_database(db_name)
    print(f"Database '{db_name}' has been deleted.")

# Create index for the vector store
def create_vector_index():
    try:
        # Create Index
        store.create_vector_search_index(
            dimensions=1024, path="embedding", similarity="cosine"
        )
        print("Vector search index created successfully.")
    except Exception as e:
        if 'An index named "vector_index" is already defined for collection embeddings.' in str(e):
            print("Vector search index already exists. Continuing...")
        else:
            raise e

# Function: `scrape_and_store_sitemap`
def scrape_and_store_sitemap(website_url: str, storage_context: StorageContext, batch_size: int = 5, limit: int = None):

    # Get links from the sitemap
    all_links = sitemap_search(website_url)
    print(f"Found {len(all_links)} links in sitemap.")

    # Limit the number of links if specified
    if limit is not None and limit > 0:
        links_to_scrape = all_links[:limit]
    else:
        links_to_scrape = all_links

    # Initialize counters and storage
    total_processed = 0
    current_batch = []
    index = None

    for i, link in enumerate(links_to_scrape):
        try:
            # Fetch the URL content
            downloaded = fetch_url(link)

            if downloaded:
                # Extract the main content
                content = extract(downloaded)

                if content:
                    # Convert each item into a Document object
                    doc = Document(text=content, metadata={"url": link})
                    current_batch.append(doc)
                    total_processed += 1

                    # Print progress
                    print(f"Processed {i+1}/{len(links_to_scrape)}: {link}")

            # Store the batch when it reaches the batch size
            if len(current_batch) >= batch_size:
                print(f"Storing batch of {len(current_batch)} documents...")

                if index is None:
                    # Create index for the first time
                    index = VectorStoreIndex.from_documents(
                        current_batch, storage_context=storage_context
                    )
                else:
                    # Update existing index with new documents
                    # Convert documents to nodes directly
                    from llama_index.core.node_parser import SentenceSplitter

                    parser = SentenceSplitter()
                    nodes = parser.get_nodes_from_documents(current_batch)
                    index.insert_nodes(nodes)

                # Clear the batch after storing
                current_batch = []
                print(f"Total documents processed so far: {total_processed}")

        except Exception as e:
            print(f"Error processing {link}: {str(e)}")

    # Store any remaining documents in the final batch
    if current_batch:
        print(f"Storing final batch of {len(current_batch)} documents...")

        if index is None:
            # Create index if this is the only batch
            index = VectorStoreIndex.from_documents(
                current_batch, storage_context=storage_context
            )
        else:
            # Update existing index with final documents
            from llama_index.core.node_parser import SentenceSplitter
            parser = SentenceSplitter()
            nodes = parser.get_nodes_from_documents(current_batch)
            index.insert_nodes(nodes)

    print(f"Completed processing. Total documents stored: {total_processed} out of {len(links_to_scrape)} links.")

# Scrape and store a website
def check_and_scrape_collection(mongodb_client, db_name, website_url, storage_context):
    db = mongodb_client[db_name]
    collection = 'embeddings'
    document_count = collection.count_documents({})

    if document_count == 0:
        # Collection is empty, proceed with scraping
        scrape_and_store_sitemap(
            website_url,
            storage_context=storage_context,
            batch_size=20,  # Process 20 documents before storing
            limit=100  #set limit to 0 to do entire website.
        )
        print("Scraping completed and data stored in the collection.")
    else:
        print(f"Collection '{collection_name}' already contains {document_count} documents. Skipping scraping process.")

##################################################
################  Front End Code #################
##################################################

@rt("/")
async def get():
    return Titled("AI Chat Assistant",
        Container(
            Card(
                Style("""
                    .chat {
                      display: flex;
                      flex-direction: column;
                      margin-bottom: 1rem;
                    }
                    .chat-header {
                      font-weight: bold;
                      margin-bottom: 0.25rem;
                    }
                    .chat-start {
                      align-items: flex-start;
                    }
                    .chat-end {
                      align-items: flex-end;
                    }
                    .chat-bubble {
                      padding: 0.75rem 1rem;
                      border-radius: 0.5rem;
                      max-width: 80%;
                    }
                    .chat-bubble-primary {
                      background-color: #3b82f6;
                      color: white;
                    }
                    .chat-bubble-secondary {
                      background-color: #e5e7eb;
                      color: #1f2937;
                    }
                """),
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
        )
    )

async def create_message_div(role, content):
    return Div(
        Div(role, cls="chat-header"),
        Div(content, cls=f"chat-bubble chat-bubble-{'primary' if role == 'user' else 'secondary'}"),
        cls=f"chat chat-{'end' if role == 'user' else 'start'}")

@rt("/send-message")
async def post_send(message: str):
    user_message = await create_message_div("user", message)
    loading_div = Div(Loading(), id="loading")
    return (
        user_message,
        TextArea(id="message", placeholder="Type your message...", value="", hx_swap_oob="true"),
        loading_div,
        Div(hx_trigger="load", hx_post="/get-response", hx_vals=f'{{"message": "{message}"}}',
            hx_target="#chat-messages", hx_swap="beforeend scroll:#chat-messages:bottom")
    )

@rt("/get-response")
async def post_response(message: str):
    try:
        ai_response = await chat_engine.query(message)
        response_div = await create_message_div("assistant", ai_response.response)
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

async def on_connect(send):
    await send(Div("Connected to chat!", cls="chat-message"))

async def on_disconnect(ws):
    print("Client disconnected")

@app.ws('/wscon', conn=on_connect, disconn=on_disconnect)
async def wscon(msg: str, send):
    await send(Div(f"Message received: {msg}", cls="chat-message"))

serve()