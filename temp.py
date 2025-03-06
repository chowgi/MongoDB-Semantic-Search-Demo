def search_bar():
    search_input = Input(type="search",
                         name="search",
                         placeholder="Search products...",
                         hx_get="/search/autocomplete",
                         hx_trigger="keyup changed delay:50ms, search",
                         hx_target="#search-results",
                         cls="search-bar")
    search_container = Container(
        id="search-results",
        cls="m-2",
        style="position: absolute; background-color:black")

    return Div(
        Grid(Div(logo, cls="col-span-1"),
             Div(search_input, search_container, cls="col-span-3"),
             cols=4,
             cls="items-center gap-4"), )


def search_products(query: str):
    """Search products using Atlas Search with fuzzy matching"""
    pipeline = [{
        "$search": {
            "index": "product_search",
            "compound": {
                "should": [{
                    "autocomplete": {
                        "query": query,
                        "path": "title",
                        "fuzzy": {
                            "maxEdits": 1,
                            "prefixLength": 0
                        },
                        "score": {
                            "boost": {
                                "value": 3
                            }
                        },
                        "tokenOrder": "sequential"
                    }
                }, {
                    "autocomplete": {
                        "query": query,
                        "path": "description",
                        "fuzzy": {
                            "maxEdits": 1,
                            "prefixLength": 0
                        },
                        "score": {
                            "boost": {
                                "value": 1
                            }
                        },
                        "tokenOrder": "sequential"
                    }
                }]
            },
            "scoreDetails": True
        }
    }, {
        "$limit": 50
    }, {
        "$project": {
            "title": 1,
            "_id": 1,
            "score": {
                "$meta": "searchScore"
            }
        }
    }, {
        "$sort": {
            "score": -1
        }
    }]
    try:
        results = list(db.products.aggregate(pipeline))
        #print(f"Search results for '{query}': {results}")
        return results
    except Exception as e:
        print(f"Atlas Search error for query '{query}': {str(e)}")
        return []


@rt("/search/autocomplete")
def get(search: str = ""):
    """Handle autocomplete requests"""
    if not search or len(search) < 2:
        return ""

    results = search_products(search)
    header = [
        Div(A(f"Search results for: {search}",
              href=f"/search/{search}",
              cls=(TextT.warning, TextT.lg)),
            cls="mb-2")
    ] if search else []

    content = [
        Div(
            A(product["title"][:80] +
              ('...' if len(product["title"]) > 80 else ''),
              href=f"/products/{product['_id']}"),
            Div(f"{product['score']:.3f}")) for product in results[:6]
    ] if results else [
        Div("No results found", style="padding: 4px 0;", cls="uk-paragraph")
    ]
    return Div(*(header + content), id="search-results")


def ex_theme_switcher():
    return ThemePicker()


test = db.runCommand({ping: 1})
print(test)


@rt("/")
def get():
    # Titled using a H1 title, sets the page title, and wraps contents in Main(Container(...)) using
    # frankenui styles. Generally you will want to use Titled for all of your pages
    return Container(Title("MongoDB Atals Search Demo"), create_header())


serve()
