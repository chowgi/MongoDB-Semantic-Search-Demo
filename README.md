
# MongoDB Semantic Search Demo

A FastHTML web application demonstrating advanced semantic search capabilities using MongoDB Atlas Vector Search and Voyage AI.

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

## Technical Stack

- **Backend Framework:** FastHTML
- **Database:** MongoDB Atlas with Vector Search
- **Embeddings:** Voyage AI (voyage-3 model)
- **Re-ranking:** Voyage AI (rerank-2 model)
- **UI Components:** MonsterUI

## Running the Application

1. Ensure all environment variables are set in Replit Secrets
2. Click the "Run" button in your Replit workspace
3. The application will start on port 5001

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
