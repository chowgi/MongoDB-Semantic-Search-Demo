# MongoDB + Voyage AI Demo

This project demonstrates MongoDB + Voyage AI integration with two standalone FastHTML applications:

1. **Search Application** - Compare Text, Vector, and Hybrid Search Methods
2. **RAG Application** - Retrieval Augmented Generation chat interface

## Environment Variables Required

Set the following environment variables before running the applications:

- `OPENAI_API_KEY` - OpenAI API key
- `MONGODB_URI` - MongoDB Atlas connection string
- `VOYAGE_API_KEY` - Voyage AI API key

## Running the Applications

### Search Application

Run the search application on port 3030:

```bash
python search_app.py
```

### RAG Application

Run the RAG application on port 3000:

```bash
python rag_app.py
```

## Tools and Libraries Used

- FastHTML - Server and UI framework
- MongoDB Atlas - Database and vector storage
- Voyage AI - Embeddings generation
- OpenAI - Language model integration