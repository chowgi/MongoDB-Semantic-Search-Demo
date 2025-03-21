# MongoDB + Voyage AI Demo

This project demonstrates MongoDB + Voyage AI integration with two standalone FastHTML applications:

1. **Search Application** - Compare Text, Vector, and Hybrid Search Methods
2. **RAG Application** - Retrieval Augmented Generation chat interface
3. **Agent Demo** - Coming Soon

## Environment Variables Required

Set the following environment variables before running the applications:

- `OPENAI_API_KEY` - OpenAI API key
- `MONGODB_URI` - MongoDB Atlas connection string
- `VOYAGE_API_KEY` - Voyage AI API key

## Running the Applications

### Search Application

Run the search application

```bash
python main.py
```

## Tools and Libraries Used

- FastHTML - Server and UI framework
- MongoDB Atlas - Database and vector storage
- Voyage AI - Embeddings generation
- OpenAI - Language model integration