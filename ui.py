
from fasthtml.common import *
from monsterui.all import *

def create_message_div(role, content):
    """Create a message div for the chat interface"""
    from fasthtml.components import NotStr
    
    # Use NotStr to allow HTML content if it's an assistant response with sources
    content_div = Div(NotStr(content) if role == "assistant" and "<br>" in str(content) else content, 
                     cls=f"chat-bubble chat-bubble-{'primary' if role == 'user' else 'secondary'}")
    
    return Div(
        Div(role, cls="chat-header"),
        content_div,
        cls=f"chat chat-{'end' if role == 'user' else 'start'}")

def navbar():
    """Create the navigation bar"""
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
    """Create the use case cards for the home page"""
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
    """Create the chat interface for the RAG page"""
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

class Loading:
    """Create a loading indicator"""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def __ft__(self):
        return Div(cls="htmx-indicator loading loading-spinner", **self.kwargs)
