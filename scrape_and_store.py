
from llama_index.core import Document
from trafilatura import fetch_url, extract
from trafilatura.sitemaps import sitemap_search
from llama_index.core import VectorStoreIndex, StorageContext
import os
import argparse
from search import init_search

def scrape_and_store_sitemap(website_url: str, storage_context: StorageContext, batch_size: int = 5, limit: int = None, collection_name: str = "embeddings"):
    """
    Scrape content from a website's sitemap and store it in a vector store.
    
    Args:
        website_url: URL of the website's sitemap
        storage_context: Storage context for the vector store
        batch_size: Number of documents to process before storing
        limit: Maximum number of links to process (None for all)
        collection_name: Name of the MongoDB collection to store embeddings in
    """
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
                    # Convert each item into a Document object with proper metadata
                    doc = Document(
                        text=content, 
                        metadata={
                            "url": link,
                            "source": "web_scrape"
                        }
                    )
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

def user_input():
    """Get user input for scraping parameters"""
    print("\n=== Web Scraping and Vector Storage Tool ===\n")
    
    # Get sitemap URL
    sitemap_url = input("Enter the sitemap URL to scrape (e.g., https://example.com/sitemap.xml): ")
    
    # Get collection name
    collection_name = input("Enter the MongoDB collection name to store embeddings (default 'embeddings'): ")
    if not collection_name:
        collection_name = "embeddings"
    
    # Get batch size
    while True:
        try:
            batch_size_input = input("Enter batch size (number of documents to process before storing, default 10): ")
            batch_size = 10 if not batch_size_input else int(batch_size_input)
            if batch_size < 1:
                print("Batch size must be at least 1. Using default of 10.")
                batch_size = 10
            break
        except ValueError:
            print("Please enter a valid number for batch size.")
    
    # Get limit
    while True:
        try:
            limit_input = input("Enter maximum number of links to process (0 for no limit): ")
            limit = 0 if not limit_input else int(limit_input)
            limit = None if limit == 0 else limit
            break
        except ValueError:
            print("Please enter a valid number for limit.")
    
    return sitemap_url, collection_name, batch_size, limit

if __name__ == "__main__":
    # Check for command line arguments, otherwise use interactive input
    parser = argparse.ArgumentParser(description='Scrape a website and store content in a vector database.')
    parser.add_argument('--sitemap', help='URL of the website sitemap')
    parser.add_argument('--collection', help='MongoDB collection name', default='embeddings')
    parser.add_argument('--batch-size', type=int, help='Batch size for processing documents', default=10)
    parser.add_argument('--limit', type=int, help='Maximum number of links to process (0 for no limit)', default=0)
    
    args = parser.parse_args()
    
    # If no sitemap URL provided in args, get inputs interactively
    if not args.sitemap:
        sitemap_url, collection_name, batch_size, limit = user_input()
    else:
        sitemap_url = args.sitemap
        collection_name = args.collection
        batch_size = args.batch_size
        limit = None if args.limit == 0 else args.limit
    
    # Check for required environment variables
    required_envs = ['MONGODB_URI', 'VOYAGE_API_KEY', 'OPENAI_API_KEY']
    missing_envs = [env for env in required_envs if not os.environ.get(env)]
    
    if missing_envs:
        print(f"Error: Missing required environment variables: {', '.join(missing_envs)}")
        print("Please set these environment variables before running the script.")
        exit(1)
    
    # Initialize search components
    mongodb_uri = os.environ['MONGODB_URI']
    voyage_api_key = os.environ['VOYAGE_API_KEY']
    openai_api_key = os.environ['OPENAI_API_KEY']
    
    print(f"\nInitializing with collection name: {collection_name}")
    print(f"Connecting to MongoDB...")
    
    search_components = init_search(
        mongodb_uri=mongodb_uri,
        voyage_api_key=voyage_api_key,
        openai_api_key=openai_api_key,
        collection_name=collection_name
    )
    
    # Set up the storage context with the MongoDB vector store
    storage_context = StorageContext.from_defaults(vector_store=search_components["store"])
    
    print(f"\nStarting to scrape {sitemap_url}")
    print(f"Batch size: {batch_size}")
    print(f"Limit: {limit if limit is not None else 'No limit'}")
    
    # Scrape and store content from the website
    scrape_and_store_sitemap(
        website_url=sitemap_url,
        storage_context=storage_context,
        batch_size=batch_size,
        limit=limit,
        collection_name=collection_name
    )
