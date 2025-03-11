
from llama_index.core import Document
from trafilatura import fetch_url, extract
from trafilatura.sitemaps import sitemap_search
from llama_index.core import VectorStoreIndex, StorageContext
import os

def scrape_and_store_sitemap(website_url: str, storage_context: StorageContext, batch_size: int = 5, limit: int = None):
    """
    Scrape content from a website's sitemap and store it in a vector store.
    
    Args:
        website_url: URL of the website's sitemap
        storage_context: Storage context for the vector store
        batch_size: Number of documents to process before storing
        limit: Maximum number of links to process (None for all)
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

if __name__ == "__main__":
    from llama_index.core import StorageContext
    import os
    from search import init_search
    
    # Initialize search components
    mongodb_uri = os.environ['MONGODB_URI']
    voyage_api_key = os.environ['VOYAGE_API_KEY']
    openai_api_key = os.environ['OPENAI_API_KEY']
    website_url = "https://www.hawthornfc.com.au/sitemap/index.xml"
    
    search_components = init_search(
        mongodb_uri=mongodb_uri,
        voyage_api_key=voyage_api_key,
        openai_api_key=openai_api_key
    )
    
    # Set up the storage context with the MongoDB vector store
    storage_context = StorageContext.from_defaults(vector_store=search_components["store"])
    
    # Scrape and store content from the website
    scrape_and_store_sitemap(
        website_url=website_url,
        storage_context=storage_context,
        batch_size=10,
        limit=20  # Set to 0 for no limit
    )
