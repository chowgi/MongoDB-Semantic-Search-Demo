from trafilatura import fetch_url, extract
from trafilatura.sitemaps import sitemap_search

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