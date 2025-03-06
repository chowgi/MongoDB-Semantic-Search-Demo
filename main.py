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

##################################################
########## Settings and Admin Logic ##############
##################################################

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

##################################################
############# Ingestion Logic ####################
##################################################

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
################# Search Logic ###################
##################################################

def search_bar():
    search_input = Input(type="search",
                         name="q",
                         placeholder="Search documents...",
                         hx_get="/search",
                         hx_target="#search-results",
                         cls="search-bar")
    search_button = Button("Search", 
                          cls=ButtonT.primary,
                          hx_get="/search",
                          hx_target="#search-results",
                          hx_include="closest div")

    # Using a Grid to place search input and button side by side
    search_form = Grid(
        Div(search_input, cls="col-span-5"),
        Div(search_button, cls="col-span-1"),
        cols=6,
        cls="items-center gap-2")

    return Div(search_form, Div(id="search-results", cls="m-2"), cls='pt-5')

def text_search(query: str):
    """Search using MongoDB Atlas Text Search with text index"""
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

def vector_search(query: str):
    """Search using MongoDB Atlas Vector Search with vector index"""
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

def hybrid_search(query: str):
    """Hybrid search using both text and vector search capabilities"""
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
################  Nav And Home ###################
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
        # Div(H4("A Simple Div with ml-20",style='background-color: red'), 
        #    cls='pt-20'),
        cls=ContainerT.sm 
    )

@rt("/search")
def get(q: str = None, request=None):
    # Check if this is an HTMX request
    is_htmx = request and request.headers.get('HX-Request') == 'true'
    search_results = Div(id="search-results", cls="m-2")

    if q and len(q) >= 2:
        # Perform all three types of searches
        text_results = text_search(q)
        vector_results = vector_search(q)
        hybrid_results = hybrid_search(q)

        # Create the comparison display
        search_results = Grid(
            Card(
                H2("Text Search", cls=TextT.primary),
                P("Traditional keyword-based search using MongoDB text index."),
                Ul(*[Li(
                    Div(result["text"][:150] + ('...' if len(result["text"]) > 150 else ''), cls="mb-2"),
                    P(f"Score: {result['score']:.3f}", cls=TextPresets.muted_sm)
                ) for result in text_results]),
                cls=CardT.primary
            ),
            Card(
                H2("Vector Search", cls=TextT.primary),
                P("Semantic search based on embeddings vector similarity."),
                Ul(*[Li(
                    Div(result["text"][:150] + ('...' if len(result["text"]) > 150 else ''), cls="mb-2"),
                    P(f"Score: {result['score']:.3f}", cls=TextPresets.muted_sm)
                ) for result in vector_results]),
                cls=CardT.primary
            ),
            Card(
                H2("Hybrid Search", cls=TextT.primary),
                P("Combined approach using both text and vector search."),
                Ul(*[Li(
                    Div(result["text"][:150] + ('...' if len(result["text"]) > 150 else ''), cls="mb-2"),
                    P(f"Score: {result['score']:.3f}", cls=TextPresets.muted_sm)
                ) for result in hybrid_results]),
                cls=CardT.primary
            ),
            cols_lg=3,
            cls="gap-4 mt-4"
        )
    
    # If it's an HTMX request, just return the search results div
    if is_htmx:
        return search_results
    
    # Otherwise, return the full page
    return Container(
        navbar(),
        Div(H2("MongoDB Atlas Search Comparison", cls="pb-10"),
            P("Compare Text, Vector, and Hybrid Search Methods", cls="pb-5"),
            search_bar()),
        search_results,
        cls=ContainerT.sm
    )

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
        ),cls=ContainerT.sm
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