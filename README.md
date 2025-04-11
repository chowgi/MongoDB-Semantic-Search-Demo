
# MongoDB Semantic Search Demo

A FastHTML web application demonstrating advanced semantic search capabilities using MongoDB Atlas Vector Search and Voyage AI.

## Try it Live!

You can "remix" the app directly on Replit here:
[https://replit.com/@BenKarciauskas/MongoDB-Semantic-Search-Demo?v=1]

## Features

- **Multi-Modal Search Comparison:**
  - Text-based search
  - Vector-based search
  - Hybrid search (combining text and vector)
  - Re-ranked vector search results

- **Interactive UI:**
  - Real-time search results
  - Search suggestions
  - Adjustable text/vector bias
  - Expandable movie plot descriptions

## Requirements

The following environment variables must be set in your Replit Secrets:

- `MONGODB_URI` - MongoDB Atlas connection string
- `VOYAGE_API_KEY` - Voyage AI API key

## Setup

1. **MongoDB Atlas Setup:**
   - Create a MongoDB Atlas account and cluster
   - In your cluster, create a database named `semantic_search_demo`
   - Create a collection named `movie_embeddings`

2. **Environment Configuration:**
   - Open the .env file
   - Add your MongoDB Atlas connection string as `MONGODB_URI`
   - Add your Voyage AI API key as `VOYAGE_API_KEY`
   - Open a shell and run "pip install -r requirements.txt"

3. **Data Loading:**
   - Run `load_data.py` to populate the database with movie data
   - This script will:
     - Load movie data from `movie_embeddings.json`
     - Create necessary search indexes for text and vector search
     - Skip existing documents to avoid duplicates

4. **Index Setup:**
   The script automatically creates two indexes:
   - A text search index for traditional text queries
   - A vector search index for semantic search with 1024-dimensional embeddings

## Technical Stack

- **Backend Framework:** FastHTML
- **Database:** MongoDB Atlas with Vector Search
- **Embeddings:** Voyage AI (voyage-3 model)
- **Re-ranking:** Voyage AI (rerank-2 model)
- **UI Components:** MonsterUI

## Running the Application

1. Ensure all environment variables are set in Replit Secrets
3. Click the "Run" button in your Replit workspace
4. The application will start on port 5001

## Project Structure

```
├── public/               # Static assets
├── main.py              # Main application code
└── requirements.txt     # Python dependencies
```

## Search Features

- **Text Search:** Traditional text-based search using MongoDB Atlas
- **Vector Search:** Semantic search using Voyage AI embeddings
- **Hybrid Search:** Combines text and vector search with adjustable weights
- **Re-ranked Search:** Enhanced vector search results using Voyage AI re-ranking

## Movie Dataset

The application searches through a movie database containing:
- Movie titles
- Ratings
- Plot descriptions
- Vector embeddings for semantic search
