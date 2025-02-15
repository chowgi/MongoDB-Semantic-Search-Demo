
# FastHTML MongoDB Template

A template for building web applications using FastHTML and MongoDB, with MonsterUI for styling.

## Features

- FastHTML for server-side rendering
- MongoDB integration
- MonsterUI components and themes
- Static file serving
- Environment variable support
- Debug mode enabled

## Getting Started

1. Set up your MongoDB connection string in the Secrets tab (Environment Variables):
   - Add `MONGO_URI` with your MongoDB connection string

2. The application will run on port 5001 and has a basic route at `/` that displays "Hello, World"

## Project Structure

```
├── static/              # Static files directory
├── main.py             # Main application file
└── pyproject.toml      # Project dependencies
```

## Dependencies

- python-fasthtml
- pymongo
- python-dotenv
- bson
- monsterui

## Usage

The template provides:

- MongoDB connection setup
- Basic FastHTML configuration with MonsterUI theme
- Static file serving
- Error handling for MongoDB connection
- Environment variable management

## Development

To modify the application:

1. Edit `main.py` to add your routes and logic
2. Add static files (CSS, JS, images) to the `static` directory
3. Use MonsterUI components for styling by importing from `monsterui.all`

## Example Route

```python
@app.get("/")
def home():
    return "<h1>Hello, World</h1>"
```

## Environment Variables

Required environment variables:
- `MONGO_URI`: Your MongoDB connection string

Set these in the Secrets tab of your Repl.

## Learn More

- [FastHTML Documentation](https://docs.fastht.ml/)
- [MonsterUI Documentation](https://monsterui.org/)
