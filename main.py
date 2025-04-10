from fasthtml.common import *
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.postprocessor.voyageai_rerank import VoyageAIRerank
from monsterui.all import *
import pymongo
import os

##################################################
########## Setting and Configuration #############
##################################################

# Initialize FastHTML with MonsterUI theme
hdrs = Theme.green.headers()
app, rt = fast_app(hdrs=hdrs, static_path="public", live=True, debug=True, title="MongoDB Semantic Search Demo")

# Retrieve environment variables for necessary API keys and URIs
mongodb_uri = os.environ['MONGODB_URI']
voyage_api_key = os.environ['VOYAGE_API_KEY']
db_name = "semantic_search_demos"
collection_name ='movie_embeddings'

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

# Set up MongoDB Atlas Vector Search connection with specified database and collection
search_store = MongoDBAtlasVectorSearch(
    mongodb_client, 
    db_name=db_name, 
    collection_name=collection_name,
    embedding_key="embedding",
    text_key="text",
    fulltext_index_name="text_index",
)

# Generate the vector index from the existing vector store
search_index = VectorStoreIndex.from_vector_store(search_store)


##################################################
##################  Functions ####################
##################################################

def navbar():
    return NavBar(brand=A(
                      DivLAligned(
                      Img(src='MongoDB.svg', height=25,width=25),
                      H4('MongoDB Semantic Search Demo')
                    ), href="/"
                 )
            )

# Build the search bar
def search_bar():
    
    # Create search suggestion buttons
    suggestions = [
        "Action movies about humans fighting robots",
        "Sci-fi films with space exploration themes",
        "Movies for kids that includes animals"
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

@rt("/")
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
            for node in nodes[:3]:

                # Extract full and truncated plot text
                full_plot = node.text.split('Plot:')[-1] if 'Plot:' in node.text else node.text
                truncated_plot = full_plot[:250] + "..." if len(full_plot) > 250 else full_plot
                
                node_content = Div(
                    P(Span("Title: ", cls="text-primary"), node.metadata['title']),
                    P(Span("Rating: ", cls="text-primary"), node.metadata['rating']),
                    P(Span("Score: ", cls="text-primary"), f"{node.score:.3f}"),
                    P(Span("Plot: ", cls="text-primary"),
                      Span(truncated_plot,
                           onclick=f"this.textContent = `{full_plot}`",
                           cls="cursor-pointer hover:text-grey-300",
                           title="Click to expand")),
                    )
                card_content.append(node_content)

            # Add the completed card with a title and content to the list
            cards.append(Card(card_title, *card_content, cls="rounded-xl"))

        grid = Div(Grid(*cards, cols_max=4, cls="gap-4"), id='search_results')  # Display in a 2-column grid

        # Return the grid first, then the clear_search_bar to ensure the results stay visible
        return grid, clear_search_bar, search_modal()
    else:
        return P("Please enter a search query.")

@rt("/suggest")
def post(query: str):
    return Input(type="search",
         name="query",
         value=query,
         placeholder="Search documents...",
         cls="search-bar",
         id="search-input",
        hx_swap_oob="true")

# Start the App
serve()