
# FastHTML MongoDB Template

A template for building web applications using FastHTML and MongoDB, with MonsterUI for styling.

## Features

- FastHTML for server-side rendering and hypermedia-driven interfaces
- MongoDB integration with automatic connection handling
- MonsterUI components and theming for beautiful UI
- Static file serving from public directory
- Environment variable support through Replit Secrets
- Debug mode enabled for development
- Automatic error handling for database connections

## Getting Started

1. Set up your MongoDB connection string in the Replit Secrets tab:
   - Add `MONGO_URI` with your MongoDB connection string

2. Run the application using the Run button - it will be available on port 5001

## Project Structure

```
├── public/              # Public assets directory
├── main.py             # Main application file 
└── pyproject.toml      # Project dependencies
```

## Dependencies

- python-fasthtml
- pymongo
- python-dotenv
- bson
- monsterui

## Template Features

The template provides:

- Configured MongoDB connection with error handling
- FastHTML setup with MonsterUI theme integration
- Public file serving 
- Environment variable management
- Basic homepage with MonsterUI Card component

## Development

To extend the application:

1. Edit `main.py` to add your routes and logic
2. Add public assets to the `public` directory
3. Use MonsterUI components for styling by importing from `monsterui.all`

## Example Route

```python
@rt("/")
def home():
    return Titled("Your App",
        Card(
            H1("Welcome!"),
            P("Start building with FastHTML and MongoDB")))
```

## Environment Variables

Required environment variable:
- `MONGO_URI`: Your MongoDB connection string

Set this in the Secrets tab of your Repl.
