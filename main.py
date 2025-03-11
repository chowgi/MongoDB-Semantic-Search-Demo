from fasthtml.common import *
from monsterui.all import *
import os
from search import init_search, search
from rag import init_rag, create_message_div, handle_chat_message

# Initialize FastHTML with MonsterUI theme
hdrs = Theme.green.headers()
app, rt = fast_app(hdrs=hdrs, static_path="public", live=True, debug=True, title="MongoDB + Voyage AI Demo")

# Retrieve environment variables for necessary API keys and URIs
openai_api_key = os.environ['OPENAI_API_KEY']
mongodb_uri = os.environ['MONGODB_URI']
voyage_api_key = os.environ['VOYAGE_API_KEY']
website_url = "https://www.hawthornfc.com.au/sitemap/index.xml"

# Initialize search components
search_components = init_search(
    mongodb_uri=mongodb_uri,
    voyage_api_key=voyage_api_key,
    openai_api_key=openai_api_key
)

# Extract components for use in the app
mongodb_client = search_components["mongodb_client"]
store = search_components["store"]
storage_context = search_components["storage_context"]
index = search_components["index"]
db_name = search_components["db_name"]

# Initialize RAG components
rag_components = init_rag(index)
chat_engine = rag_components["chat_engine"]

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
        hx_indicator="#loading"
    )

    return Div(search_form, cls='pt-5')

# Search function is now imported from search.py  

##################################################
################## RAG Logic #####################
##################################################

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
        # Use the imported search function from search.py
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


@rt("/rag")
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
    ),Div(Loading(cls=LoadingT.dots), id="loading")

@rt("/get-response")
def post(message: str):
    # Use the RAG module to handle the chat message
    ai_response = handle_chat_message(chat_engine, message)

    return (
        create_message_div("assistant", ai_response),
        Div(id="loading", hx_swap_oob="true"))


serve()