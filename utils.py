
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Document
from trafilatura import fetch_url, extract
from trafilatura.sitemaps import sitemap_search
import pymongo
import os

def initialize_settings(openai_api_key, voyage_api_key):
    """Configure the default LLM and embedding model settings"""
    Settings.llm = OpenAI(
        temperature=0.7, model="gpt-3.5-turbo", api_key=openai_api_key
    )
    
    Settings.embed_model = VoyageEmbedding(
        voyage_api_key=voyage_api_key,
        model_name="voyage-3",
    )

def get_mongodb_client(mongodb_uri):
    """Establish MongoDB client connection"""
    return pymongo.MongoClient(mongodb_uri)

def delete_db(mongodb_client, db_name):
    """Delete a MongoDB database"""
    mongodb_client.drop_database(db_name)
    print(f"Database '{db_name}' has been deleted.")

def create_vector_index(store):
    """Create index for the vector store"""
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

def scrape_and_store_sitemap(website_url, storage_context, batch_size=5, limit=None):
    """Scrape a website sitemap and store content in vector index"""
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

def check_and_scrape_collection(mongodb_client, db_name, website_url, storage_context, collection='embeddings'):
    """Check if collection exists and scrape if empty"""
    db = mongodb_client[db_name]
    document_count = db[collection].count_documents({})

    if document_count == 0:
        # Collection is empty, proceed with scraping
        scrape_and_store_sitemap(
            website_url,
            storage_context=storage_context,
            batch_size=20,  # Process 20 documents before storing
            limit=100  # Set limit to 0 to do entire website
        )
        print("Scraping completed and data stored in the collection.")
    else:
        print(f"Collection '{collection}' already contains {document_count} documents. Skipping scraping process.")

def setup_vector_search(mongodb_uri, db_name, website_url):
    """Set up MongoDB Atlas Vector Search and scrape website if needed"""
    # Establish MongoDB client connection
    mongodb_client = get_mongodb_client(mongodb_uri)
    
    # Set up MongoDB Atlas Vector Search connection
    store = MongoDBAtlasVectorSearch(mongodb_client, db_name=db_name, collection_name='embeddings')
    
    # Initialize the storage context
    storage_context = StorageContext.from_defaults(vector_store=store)
    
    # Create vector index if needed
    create_vector_index(store)
    
    # Check if collection exists and scrape if empty
    check_and_scrape_collection(mongodb_client, db_name, website_url, storage_context)
    
    # Generate the vector index from the existing vector store
    index = VectorStoreIndex.from_vector_store(store)
    
    # Create chat engine
    chat_engine = index.as_query_engine(similarity_top_k=3)
    
    return mongodb_client, store, index, chat_engine
