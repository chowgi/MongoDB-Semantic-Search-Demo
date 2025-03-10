from fasthtml.common import *
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.llms.openai import OpenAI
from monsterui.all import *
import pymongo
import os

# Initialize FastHTML with MonsterUI theme
hdrs = Theme.green.headers()
app, rt = fast_app(hdrs=hdrs, static_path="public", live=True, debug=True, title="MongoDB + Voyage AI Demo")

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
store = MongoDBAtlasVectorSearch(
    mongodb_client, 
    db_name=db_name, 
    collection_name='embeddings', #<--- do I need this?
    embedding_key="embedding",
    text_key="text",
    fulltext_index_name="text_index",
)

# Initialize the storage context for vector store operations
storage_context = StorageContext.from_defaults(vector_store=store)

# Generate the vector index from the existing vector store
index = VectorStoreIndex.from_vector_store(store)

# create the chat engine
chat_engine = index.as_query_engine(similarity_top_k=3)


def TicketSteps(step):
    return Steps(
        LiStep("Initial Planning", data_content="📝",
               cls=StepT.success if step > 0 else StepT.primary if step == 0 else StepT.neutral),
        LiStep("Development", data_content="🔎",
               cls=StepT.success if step > 1 else StepT.primary if step == 1 else StepT.neutral),
        LiStep("Testing", data_content="⚙️",
               cls=StepT.success if step > 2 else StepT.primary if step == 2 else StepT.neutral),
        LiStep("Release", data_content="✅",
               cls=StepT.success if step > 3 else StepT.primary if step == 3 else StepT.neutral),
        cls="w-full")


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
        hx_indicator="#search-loading"
    )
    
    loading = Loading(htmx_indicator=True, cls=LoadingT.dots, id="search-loading")

    return Div(search_form, loading, cls='pt-5')

def search(query, top_k=5):
    modes = ["text_search", "default", "hybrid"]  # default is vector
    results = {}  # Initialize results as an empty dictionary

    for mode in modes:
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

    return results  

##################################################
################## RAG Logic #####################
##################################################

def create_message_div(role, content):
    icon = UkIcon("user" if role == "user" else "robot")
    
    if role == "user":
        return Div(
            DivRAligned(
                Card(
                    DivLAligned(icon, Strong("You"), cls="mb-2"),
                    P(content, cls=TextT.sm),
                    cls=CardT.primary
                ),
                cls="w-3/4"
            ),
            cls="mb-4"
        )
    else:
        return Div(
            DivLAligned(
                Card(
                    DivLAligned(icon, Strong("Assistant"), cls="mb-2"),
                    Div(content, cls=(TextT.sm, "prose")),
                    cls=CardT.secondary
                ),
                cls="w-3/4"
            ),
            cls="mb-4"
        )

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
    return Title("MongoDB + Voyage AI Demos"), Container(
        navbar(),
        use_case_cards(),
        cls=ContainerT.lg
    )

@rt("/search")
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
        results = search(query, top_k=3)

        # Mode descriptions and alert styles
        mode_info = {


@rt("/update-settings")
def post(mongodb_uri: str = None, db_name: str = None, collection_name: str = None, 
         voyage_api_key: str = None, embedding_model: str = None, website_url: str = None):
    # In a production app, you would update environment variables or a configuration file
    # For this demo, we'll just return a success message
    
    return Alert(
        DivLAligned(UkIcon("check"), Span("Settings updated successfully!")),
        cls=(AlertT.success, "mb-4"),
        hx_swap_oob="true"
    )

@rt("/reset-vector-store")
def post():
    # In a production app, this would delete the MongoDB collection and recreate it
    try:
        # Simulate database reset with a delay
        import time
        time.sleep(2)
        
        return Alert(
            DivLAligned(UkIcon("check"), Span("Vector store reset successfully!")),
            cls=(AlertT.success, "mb-4"),
            hx_swap_oob="true"
        )
    except Exception as e:
        return Alert(
            DivLAligned(UkIcon("warning"), Span(f"Error: {str(e)}")),
            cls=(AlertT.error, "mb-4"),
            hx_swap_oob="true"
        )


            "Text Search": {
                "description": "Uses traditional keyword-based searching to find exact matches in text.",
                "alert_type": AlertT.info
            },
            "Vector Search": {
                "description": "Uses AI embeddings to find semantically similar content regardless of wording.",
                "alert_type": AlertT.success 
            },
            "Hybrid Search": {
                "description": "Combines text and vector search for both exact and semantic matches.",
                "alert_type": AlertT.warning
            }
        }

        # Create a card for each mode with the mode_name as the title
        cards = []  # Initialize the cards list
        for mode_name, nodes in results.items(): 
            info = mode_info.get(mode_name, {"description": "Search mode", "alert_type": AlertT.info})
            
            card_header = Div(
                H4(mode_name),
                Alert(info["description"], cls=info["alert_type"]),
                cls="space-y-2"
            )
            
            card_content = []
            for i, node in enumerate(nodes):
                # Format the score as percentage for better readability
                score_percent = f"{node.score * 100:.1f}%"
                
                node_content = Card(
                    P(node.node.text[:200] + "..."),
                    DividerSplit(),
                    DivFullySpaced(
                        Div(Strong("Relevance: "), score_percent),
                        Div(
                            A(
                                UkIcon("external-link", cls="mr-1"),
                                "Source",
                                href=node.metadata['url'],
                                target='_blank',
                                cls="text-primary"
                            )
                        )
                    ),
                    cls=CardT.hover
                )
                card_content.append(node_content)

            # Add the completed card with a title and content to the list
            cards.append(Card(card_header, *card_content))

        grid = Grid(*cards, cols_lg=3, cls="gap-4")  # Display in a 3-column grid

        return clear_search_bar, grid
    else:
        return P("Please enter a search query.")


@rt("/rag")
def get():
    return Container(
        navbar(),
        Card(
            H3("RAG Chat with MongoDB Atlas & Voyage AI", cls="text-center mb-4"),
            Alert(
                DivLAligned(UkIcon("info"), Span("This chat uses MongoDB Atlas and Voyage AI to retrieve and summarize relevant information based on your questions.")),
                cls=(AlertT.info, "mb-4")
            ),
            Div(id="chat-messages", 
                cls="space-y-4 overflow-y-auto p-4 rounded-lg bg-gray-50",
                style="height:400px; overflow: auto"
            ),
            Form(
                TextArea(id="message", placeholder="Ask a question about the website content...", cls="w-full"),
                DivRAligned(
                    Button(
                        DivLAligned(UkIcon("send"), "Send"),
                        cls=ButtonT.primary,
                        hx_post="/send-message",
                        hx_target="#chat-messages",
                        hx_swap="beforeend scroll:#chat-messages:bottom",
                        hx_indicator="#rag-loading"
                    )
                ),
                cls="space-y-2",
                hx_trigger="keydown[key=='Enter' && !shiftKey]",
                hx_post="/send-message",
                hx_target="#chat-messages",
                hx_swap="beforeend scroll:#chat-messages:bottom",
                hx_indicator="#rag-loading"
            ),
            Loading(htmx_indicator=True, type=LoadingT.dots, cls="fixed top-0 right-0 m-4", id="rag-loading"),
            cls=(CardT.hover, "shadow-lg")
        ),
        cls=ContainerT.lg
    )

@rt("/agents")
def get():
    return Container(
        navbar(),
        Card(
            H3("AI Agents (Coming Soon)", cls="mb-4"),
            Alert(
                DivLAligned(UkIcon("code"), Span("This feature is under development")),
                cls=(AlertT.warning, "mb-4")
            ),
            Grid(
                Card(
                    Img(src="/ai_icon.png", alt="Data Analysis Agent", style="height:100px; object-fit:contain; margin:auto;"),
                    H4("Data Analysis Agent"),
                    P("Process and analyze your MongoDB data with natural language requests"),
                    Button("Try Demo", cls=ButtonT.primary, disabled=True),
                    cls=CardT.hover
                ),
                Card(
                    Img(src="/ai_icon.png", alt="Research Assistant", style="height:100px; object-fit:contain; margin:auto;"),
                    H4("Research Assistant"),
                    P("Automatically collect and synthesize information from your knowledge base"),
                    Button("Try Demo", cls=ButtonT.primary, disabled=True),
                    cls=CardT.hover
                ),
                Card(
                    Img(src="/ai_icon.png", alt="Workflow Automation", style="height:100px; object-fit:contain; margin:auto;"),
                    H4("Workflow Automation"),
                    P("Create and execute multi-step workflows with MongoDB integrations"),
                    Button("Try Demo", cls=ButtonT.primary, disabled=True),
                    cls=CardT.hover
                ),
                cols_lg=3,
                cls="gap-4"
            ),
            DividerSplit("Feature Roadmap"),
            TicketSteps(1),
            cls=(CardT.hover, "shadow-lg")
        ),
        cls=ContainerT.lg
    )

@rt("/settings")
def get():
    return Container(
        navbar(),
        Card(
            H3("MongoDB Atlas Settings", cls="mb-4"),
            Form(
                LabelInput("MongoDB URI", id="mongodb_uri", type="password", value=mongodb_uri if mongodb_uri else "", placeholder="mongodb+srv://..."),
                LabelInput("Database Name", id="db_name", value=db_name if db_name else "", placeholder="your_database_name"),
                LabelInput("Collection Name", id="collection_name", value="embeddings", placeholder="embeddings"),
                LabelInput("Voyage AI API Key", id="voyage_api_key", type="password", value=voyage_api_key if voyage_api_key else "", placeholder="Enter your Voyage AI API key"),
                H4("Indexing Settings", cls="mt-6 mb-2"),
                Grid(
                    LabelInput("Website URL", id="website_url", value=website_url if website_url else "", placeholder="https://example.com/sitemap.xml"),
                    LabelInput("Batch Size", id="batch_size", type="number", value="20", placeholder="20"),
                    LabelSelect(*Options("Embedding Model", "voyage-2", "voyage-3", selected_idx=2), label="Embedding Model", id="embedding_model"),
                    LabelInput("Scrape Limit", id="scrape_limit", type="number", value="100", placeholder="100 (0 for all)"),
                    cols=2
                ),
                DividerSplit(),
                DivRAligned(
                    Button("Save Settings", cls=ButtonT.primary, hx_post="/update-settings", hx_swap="outerHTML"),
                    Button("Reset Vector Store", cls=(ButtonT.destructive, "ml-2"), hx_post="/reset-vector-store", hx_confirm="This will delete all embeddings. Are you sure?", hx_swap="none")
                ),
                cls="space-y-4"
            )
        ),
        cls=ContainerT.lg
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