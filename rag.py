
from llama_index.core import VectorStoreIndex
from fasthtml.common import *

def init_rag(index):
    """Initialize RAG components and return the configured chat engine"""
    
    # Create the chat engine with reasonable defaults
    chat_engine = index.as_query_engine(similarity_top_k=3)
    
    return {
        "chat_engine": chat_engine
    }

def create_message_div(role, content):
    """Create a chat message div with proper styling"""
    return Div(
        Div(role, cls="chat-header"),
        Div(content, cls=f"chat-bubble chat-bubble-{'primary' if role == 'user' else 'secondary'}"),
        cls=f"chat chat-{'end' if role == 'user' else 'start'}")

def handle_chat_message(chat_engine, message):
    """Process a chat message and generate a response using the RAG system"""
    ai_response = chat_engine.query(message)
    return ai_response
