from fasthtml.common import *
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.postprocessor.voyageai_rerank import VoyageAIRerank
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer
from monsterui.all import *
import pymongo
import os

##################################################
################# Global Settings ################
##################################################

# Initialize FastHTML with MonsterUI theme
hdrs = Theme.green.headers()
app, rt = fast_app(hdrs=hdrs, static_path="public", live=True, debug=True, title="MongoDB + Voyage AI Demo")

# Retrieve environment variables for necessary API keys and URIs
openai_api_key = os.environ['OPENAI_API_KEY']
mongodb_uri = os.environ['MONGODB_URI']
voyage_api_key = os.environ['VOYAGE_API_KEY']
db_name = "mongo_voyage_demos"

# Configure the default Language Model with OpenAI's API
Settings.llm = OpenAI(
    temperature=0.7, model="gpt-3.5-turbo", api_key=openai_api_key
)

# Set the default VoyageAI Embedding and re-ranker
Settings.embed_model = VoyageEmbedding(
    voyage_api_key=voyage_api_key,
    model_name="voyage-3",
)

voyageai_rerank = VoyageAIRerank(
    api_key=voyage_api_key, top_k=5, model="rerank-2", truncation=True
)

# Establish MongoDB client connection using the provided URI
mongodb_client = pymongo.MongoClient(mongodb_uri)

##################################################
################  Landing Page ###################
##################################################f

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

@rt("/")
def get():
    return Title("MongoDB + Voyage AI Demos"), Container(
        navbar(),
        use_case_cards(),
        cls=ContainerT.lg
    )

##################################################
#################### Search ######################
##################################################

# Set up MongoDB Atlas Vector Search connection with specified database and collection
search_store = MongoDBAtlasVectorSearch(
    mongodb_client, 
    db_name=db_name, 
    collection_name='movie_embeddings',
    embedding_key="embedding",
    text_key="text",
    fulltext_index_name="text_index",
)

# Generate the vector index from the existing vector store
search_index = VectorStoreIndex.from_vector_store(search_store)

# Build the search bar
def search_bar():
    
    # Create search suggestion buttons
    suggestions = [
        "Action movies about humans fighting robots",
        "Sci-fi films with space exploration themes",
        "Crime thrillers with unexpected plot twists"
    ]

    suggestion_buttons = []
    for suggestion in suggestions:
        suggestion_buttons.append(
            Button(suggestion,
                   name="query",
                   cls="text-sm hover:bg-gray-700 hover:text-white rounded mb-2 mr-2",
                   hx_target="#search-input",
                   hx_post=f"/suggest?query={suggestion}",
                   hx_swap="OuterHTML"
                  )
        )

    # Create suggestion container
    suggestion_container = Div(
        P("Try these searches:", cls="font-bold mb-2"),
        DivHStacked(*suggestion_buttons, cls="flex-wrap"),
        cls="mb-4"
    )

    search_input = Input(type="search",
                         name="query",
                         placeholder="Search documents...",
                         cls="search-bar",
                         id="search-input")
    search_button = Button("Search", 
                          cls=ButtonT.primary,
                          type="submit")

    alpha_range = Range(value='5', min='1', max='10', name='alpha', id='alpha')

    search_form = Form(
        Grid(
            Div(search_input, cls="col-span-7"),
            Div(search_button, cls="col-span-1"),
            Div(P("Text/Vector Bias:", cls="col-span-2")),
            Div(alpha_range, cls="col-span-2"),
            cols=8,
            cls="items-center gap-2"
        ),
        hx_get="/search/results",
        hx_target="#search-results",
        hx_trigger="submit, keyup[key=='Enter'] from:input[name='query']",
        hx_indicator="#loading"
    )
    search_input = Card(
        suggestion_container,
        search_form,
        cls="rounded-xl"
    )

    return Div(search_input, cls='pt-5')

def search(query, alpha):
    modes = ["text_search", "default", "hybrid"]  # default is vector
    results = {}  # Initialize results as an empty dictionary
    for mode in modes:
        # Create a retriever with the specific mode
        retriever = search_index.as_retriever(
            similarity_top_k=5,
            vector_store_query_mode=mode,
            alpha=alpha
        )

        # Retrieve nodes using the current mode
        retrieved_nodes = retriever.retrieve(query)

        # Map modes to titles for clarity
        mode_name = "Text Search" if mode == "text_search" else ("Vector Search" if mode == "default" else "Hybrid Search")

        # Store the retrieved nodes in the results dictionary using mode_name as key
        results[mode_name] = retrieved_nodes

    # Setup separate query engine to enable re-ranking of results
    query_engine = search_index.as_query_engine(
        similarity_top_k=5,
        node_postprocessors=[voyageai_rerank],
        alpha=alpha
    )

    retrieved_nodes = query_engine.query(query)

    # Add re-ranked results to the results dictionary
    results["Re-ranked Vector Search"] = retrieved_nodes.source_nodes

    return results  

def search_modal():
    return DivCentered(
        Button("Show me whats going on", data_uk_toggle="target: #my-modal"),
        Modal(
            ModalTitle("Search Demo Diagram"),
            Img(src="/search_diagram.png",
                alt="Search Demo Diagram",
                style="width:100%; height:auto; display:block; margin:auto;"),
            footer=ModalCloseButton("Close", cls=ButtonT.primary),
            id='my-modal'
        ),
        cls="mt-5"
    )

@rt("/search")
def get():
    """Main search page that displays the search form and empty results container"""
    search_results = Div(id="search-results", cls="m-2")

    return Title("Search - MongoDB + Voyage AI"), Container(
        navbar(),
        Div(H2("Movie Search", cls="pb-10 text-center"),
            P("Compare Text, Vector, Hybrid and Re-ranked Search Methods", cls="pb-5 text-center uk-text-lead"),
            search_bar(),
            cls="container mx-auto p-4"),
        Div(
            Div(P("Searching", cls="mr-2"),
                Loading(cls=LoadingT.dots), 
                cls="flex items-center justify-center"),
            id="loading", 
            cls="htmx-indicator flex items-center justify-center h-12"
        ),
        Div(id="search-results", cls="m-2"),
        cls=ContainerT.lg
    )


@rt("/search/results")
def get(query: str, alpha: int):

    clear_search_bar = Input(type="search",
         name="query",
         placeholder="Search documents...",
         cls="search-bar",
         id="search-input",
         hx_swap_oob="true")

    if query:
        results = search(query, alpha=alpha/10)

        # Create a card for each mode with the mode_name as the title
        cards = []  # Initialize the cards list
        for mode_name, nodes in results.items(): 

            card_title = H4(f'{mode_name}')
            card_content = []
            for node in nodes:

                # Get plot text and create truncated version
                plot_text = node.metadata.get('text', '')
                truncated_text = plot_text[:100] + '...' if len(plot_text) > 100 else plot_text
                
                node_content = Div(
                    P(Span("Title: ", cls="text-primary"), node.metadata['title']),
                    P(Span("Rating: ", cls="text-primary"), node.metadata['rating']),
                    P(Span("Score: ", cls="text-primary"), f"{node.score:.3f}"),
                    P(Span("Plot: ", cls="text-primary"),
                      Div(truncated_text, 
                          cls="truncated-text cursor-pointer",
                          hx_get=f"/expand-text?text={plot_text}",
                          hx_target="closest div",
                          hx_swap="outerHTML")),
                    )
                card_content.append(node_content)

            # Add the completed card with a title and content to the list
            cards.append(Card(card_title, *card_content, cls="rounded-xl"))

        grid = Div(Grid(*cards, cols_lg=3, cls="gap-4"), id='search_results')  # Display in a 3-column grid

        # Return the grid first, then the clear_search_bar to ensure the results stay visible
        return grid, clear_search_bar, search_modal()
    else:
        return P("Please enter a search query.")

@rt("/expand-text")
def get(text: str):
    return Div(text, 
               cls="expanded-text cursor-pointer",
               hx_get=f"/collapse-text?text={text[:100]}",
               hx_target="closest div",
               hx_swap="outerHTML")

@rt("/collapse-text")
def get(text: str):
    return Div(f"{text}...", 
               cls="truncated-text cursor-pointer",
               hx_get=f"/expand-text?text={text}",
               hx_target="closest div",
               hx_swap="outerHTML")

@rt("/suggest")
def post(query: str):
    return Input(type="search",
         name="query",
         value=query,
         placeholder="Search documents...",
         cls="search-bar",
         id="search-input",
        hx_swap_oob="true")


##################################################
####################### RAG ######################
##################################################

rag_collection_name = 'bendigo_embeddings'

# Set up MongoDB Atlas Vector Search connection with specified database and collection
rag_store = MongoDBAtlasVectorSearch(
    mongodb_client, 
    db_name=db_name, 
    collection_name=rag_collection_name,
    embedding_key="embedding"
)

## Initialize the storage context for vector store operations
#rag_storage_context = StorageContext.from_defaults(vector_store=search_store)

# Generate the vector index from the existing vector store
rag_index = VectorStoreIndex.from_vector_store(rag_store)

# Configure memory as a test. Need to implment in mongo. 
memory = ChatMemoryBuffer.from_defaults(token_limit=500)


# Function to create a chat engine
def create_chat_engine(use_rerank):


    node_postprocessors = [voyageai_rerank] if use_rerank else []
    print(f"Reranking used in chat_engine: {use_rerank}")

    return rag_index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        node_postprocessors=node_postprocessors,
        similarity_top_k=3,
        context_prompt=(
            "You are a Bendigo Bank assistant who can answer questions about Bendigo Bank products, services and anything else related to the bank. If the answer is not contained in the above context just say 'I don't know' "),
        verbose=True,
    )

# Function to create a chat engine
def create_query_engine(use_rerank):


    node_postprocessors = [voyageai_rerank] if use_rerank else []
    print(f"Reranking used in chat_engine: {use_rerank}")

    return rag_index.as_query_engine(
        node_postprocessors=node_postprocessors,
        similarity_top_k=3,
        verbose=True,
    )


def create_message_div(role, content):
    source_divs = []
    if role == "assistant" and hasattr(content, 'source_nodes'):
        sources = get_sources(content)
        for source in sources:
            source_divs.append(Div(source, cls="chat-source"))
        return Div(
            Div(role, cls="chat-header"),
            Div(str(content), P("Sources", cls="mt-2"), *source_divs, cls=f"chat-bubble chat-bubble-secondary"),
            cls="chat chat-start")
    return Div(
        Div(role, cls="chat-header"),
        Div(str(content), cls=f"chat-bubble chat-bubble-{'primary' if role == 'user' else 'secondary'}"),
        cls=f"chat chat-{'end' if role == 'user' else 'start'}")


def get_sources(ai_response):
    sources = []
    for node in ai_response.source_nodes:
        if 'url' in node.node.metadata:
            url = node.node.metadata['url']
            score = node.score
            sources.append(P(
                A(url, href=url, target="_blank", cls=AT.muted),
                Span(f" (Score: {score:.3f})", cls=AT.muted)
            ))
    return sources


def rag_suggestions():
    suggestions = [
        "What would be a good product for my young son?",
        "What are your best home loan rates?",
        "Can you tell me about NAB term deposit products?"
    ]

    suggestion_buttons = []
    for suggestion in suggestions:
        suggestion_buttons.append(
            Button(suggestion,
                   cls="text-sm hover:bg-gray-700 hover:text-white rounded mb-2 mr-2",
                   hx_post="/send-message",
                   hx_vals=f'{{"message": "{suggestion}"}}',
                   hx_target="#chat-messages",
                   hx_swap="beforeend scroll:#chat-messages:bottom")
        )

    return Div(
        DivHStacked(P("Try these questions:    ", cls="font-bold mr-4"), *suggestion_buttons, cls="flex-wrap")
    )

def chatbot_interface():
    chat_messages = Card(
            Div(id="chat-messages", 
                cls="space-y-4 h-[60vh] overflow-y-auto p-4",
                style="overflow: auto"
               ),
        cls="mb-4 rounded-xl"
    )

    chat_input = TextArea(id="message", placeholder="How can I help you today?")
    chat_button = Button("Ask",
        cls=ButtonT.primary,
        hx_post="/send-message",
        hx_target="#chat-messages",
        hx_swap="beforeend scroll:#chat-messages:bottom"
    )

    chat_form = Card(
        Form(
            rag_suggestions(),
            Grid(
                Div(chat_input, cls="col-span-7"),
                Div(chat_button, cls="col-span-1"),
                cols=8,
                cls="items-center gap-2"
            ),
            DivHStacked(
                Switch(name="use_rerank", id="use_rerank"),
                P("Use VoyageAI Reranking"),
                cls="flex items-center gap-2 mt-2"
            ),
            cls="space-y-2",
            hx_trigger="keydown[key=='Enter' && !shiftKey]",
            hx_post="/send-message",
            hx_target="#chat-messages",
            hx_swap="beforeend scroll:#chat-messages:bottom"
        ),
        cls="rounded-xl"
    )
    return chat_messages, chat_form

@rt("/rag")
def get():
    return Title("MongoDB + Voyage AI Demo"), Container(
        navbar(),
        Div(H2("Q&A With Your Data", cls="pb-10 text-center"),
            P("Deliver contextually relevant and accurate responses based on private data sources. Commonly referred to as Resource Augmented Generation (RAG).", cls="pb-5 text-center uk-text-lead")),
        chatbot_interface(),
        cls=ContainerT.lg
    )

@rt("/send-message")
def post(message: str, use_rerank: bool = False):

    # Use the boolean value directly
    #rerank_value = use_rerank

    return (
        create_message_div("user", message),
        TextArea(id="message", placeholder="Type your message...", hx_swap_oob="true"),
        Div(hx_trigger="load", 
            hx_post="/get-response", 
            hx_vals=f'{{"message": "{message}", "use_rerank": "{use_rerank}"}}',
            hx_target="#chat-messages", 
            hx_swap="beforeend scroll:#chat-messages:bottom")
    ), Div(Loading(cls=LoadingT.dots), id="loading")


@app.post("/get-response")
def post(message: str, use_rerank: bool):
    try:

        # chat_engine = create_chat_engine(use_rerank)
        # ai_response = chat_engine.chat(message)

        query_engine = create_query_engine(use_rerank)
        ai_response = query_engine.query(message)

        return (
            create_message_div(
                "assistant",
                ai_response,
            ),
            Div(id="loading", hx_swap_oob="true"))
    except Exception as e:
        print(f"Error in get-response: {str(e)}")
        # Return a graceful error message
        return (
            create_message_div(
                "assistant",
                "I'm sorry, I encountered an error processing your request. Please try again."
            ),
            Div(id="loading", hx_swap_oob="true"))




##################################################
#################### Agents ######################
##################################################

@rt("/agents")
def get():
    return Container(
        navbar(),
        P("Agent demo coming soon! Maybe..."),
        cls=ContainerT.sm
    )

#################################################
################## Settings ######################
##################################################

@rt("/settings")
def get():
    return Container(
        navbar(),
        P("Settings coming soon! Maybe..."),
        cls=ContainerT.sm
    )

# Seart the App
serve()