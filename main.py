
###
# Walkthrough of an idiomatic fasthtml app with MongoDB
###

# This fasthtml app includes functionality from fastcore, starlette, and fasthtml itself.
# Importing from `fasthtml.common` brings the key parts of all of these together.
from fasthtml.common import *
from hmac import compare_digest
from pymongo import MongoClient
from bson import ObjectId
import os
from dotenv import load_dotenv

# Load environment variables for MongoDB connection
load_dotenv()

# You can use any database you want - here we're using MongoDB which is great for document storage
# MongoDB automatically creates databases and collections when they're first used
client = MongoClient(os.environ['MONGO_URI'])
db = client.fasthtml_db
todos_collection = db.todos
users_collection = db.users

# Create indexes for better query performance
todos_collection.create_index([("name", 1)])
users_collection.create_index([("name", 1)], unique=True)

# Status code 303 is a redirect that can change POST to GET, so it's appropriate for a login page
login_redir = RedirectResponse('/login', status_code=303)

# The `before` function is a *Beforeware* function. These are functions that run before a route handler is called.
def before(req, sess):
    # This sets the `auth` attribute in the request scope, and gets it from the session
    auth = req.scope['auth'] = sess.get('auth', None)
    # If the session key is not there, it redirects to the login page
    if not auth: return login_redir

markdown_js = """
import { marked } from "https://cdn.jsdelivr.net/npm/marked/lib/marked.esm.js";
proc_htmx('.markdown', e => e.innerHTML = marked.parse(e.textContent));
"""

def _not_found(req, exc): return Titled('Oh no!', Div('We could not find that page :('))

bware = Beforeware(before, skip=[r'/favicon\.ico', r'/static/.*', r'.*\.css', '/login'])
app = FastHTML(before=bware,
               exception_handlers={404: _not_found},
               hdrs=(picolink,
                     Style(':root { --pico-font-size: 100%; }'),
                     SortableJS('.sortable'),
                     Script(markdown_js, type='module')))
rt = app.route

@rt("/login")
def get():
    frm = Form(
        Input(id='name', placeholder='Name'),
        Input(id='pwd', type='password', placeholder='Password'),
        Button('login'),
        action='/login', method='post')
    return Titled("Login", frm)

@rt("/login")
def post(request):
    form_data = dict(request.form)
    if not form_data.get('name') or not form_data.get('pwd'): return login_redir

    # Find or create user in MongoDB
    user = users_collection.find_one({"name": form_data['name']})
    if not user:
        users_collection.insert_one({"name": form_data['name'], "pwd": form_data['pwd']})
        user = {"name": form_data['name'], "pwd": form_data['pwd']}

    if not compare_digest(user['pwd'].encode("utf-8"), form_data['pwd'].encode("utf-8")): 
        return login_redir

    request.session['auth'] = user['name']
    return RedirectResponse('/', status_code=303)

@app.get("/logout")
def logout(sess):
    del sess['auth']
    return login_redir

@rt("/{fname:path}.{ext:static}")
def get(fname:str, ext:str): return FileResponse(f'{fname}.{ext}')

def todo_to_ft(todo):
    """Convert a MongoDB todo document to FastHTML components"""
    show = AX(todo['title'], f'/todos/{str(todo["_id"])}', 'current-todo')
    edit = AX('edit', f'/edit/{str(todo["_id"])}', 'current-todo')
    dt = '✅ ' if todo.get('done', False) else ''
    cts = (dt, show, ' | ', edit, Hidden(id="id", value=str(todo["_id"])), Hidden(id="priority", value=todo.get("priority", 0)))
    return Li(*cts, id=f'todo-{str(todo["_id"])}')

@rt("/")
def get(auth):
    title = f"{auth}'s Todo list"
    top = Grid(H1(title), Div(A('logout', href='/logout'), style='text-align: right'))
    new_inp = Input(id="new-title", name="title", placeholder="New Todo")
    add = Form(Group(new_inp, Button("Add")),
               hx_post="/", target_id='todo-list', hx_swap="afterbegin")

    # Query MongoDB for todos, sorted by priority
    todos = [todo_to_ft(t) for t in todos_collection.find({"name": auth}).sort("priority")]
    frm = Form(*todos, id='todo-list', cls='sortable', hx_post="/reorder", hx_trigger="end")
    card = Card(Ul(frm), header=add, footer=Div(id='current-todo'))
    return Title(title), Container(top, card)

@rt("/reorder")
def post(request):
    id_list = request.form.getlist("id")
    # Update priorities in MongoDB
    for i, id_ in enumerate(id_list):
        todos_collection.update_one({"_id": ObjectId(id_)}, {"$set": {"priority": i}})
    auth = request.scope['auth']
    todos = [todo_to_ft(t) for t in todos_collection.find({"name": auth}).sort("priority")]
    return tuple(todos)

def clr_details(): return Div(hx_swap_oob='innerHTML', id='current-todo')

@rt("/todos/{id}")
def delete(id:str):
    todos_collection.delete_one({"_id": ObjectId(id)})
    return clr_details()

@rt("/edit/{id}")
def get(id:str):
    todo = todos_collection.find_one({"_id": ObjectId(id)})
    res = Form(Group(Input(id="title", value=todo['title']), Button("Save")),
        Hidden(id="id", value=str(todo['_id'])), 
        CheckboxX(id="done", label='Done', checked=todo.get('done', False)),
        Textarea(id="details", name="details", rows=10, value=todo.get('details', '')),
        hx_put="/", target_id=f'todo-{id}', id="edit")
    return res

@rt("/")
def put(request):
    form_data = dict(request.form)
    todo_id = ObjectId(form_data['id'])
    update_data = {
        "title": form_data['title'],
        "done": form_data.get('done') == 'on',
        "details": form_data.get('details', '')
    }
    todos_collection.update_one({"_id": todo_id}, {"$set": update_data})
    updated = todos_collection.find_one({"_id": todo_id})
    return todo_to_ft(updated), clr_details()

@rt("/")
async def post(request):
    form = await request.form()
    new_inp = Input(id="new-title", name="title", placeholder="New Todo", hx_swap_oob='true')
    auth = request.scope['auth']
    new_todo = {
        "title": form['title'],
        "done": False,
        "name": auth,
        "details": "",
        "priority": 0
    }
    result = todos_collection.insert_one(new_todo)
    inserted = todos_collection.find_one({"_id": result.inserted_id})
    return todo_to_ft(inserted), new_inp

@rt("/todos/{id}")
def get(id:str):
    todo = todos_collection.find_one({"_id": ObjectId(id)})
    btn = Button('delete', hx_delete=f'/todos/{id}',
                 target_id=f'todo-{id}', hx_swap="outerHTML")
    return Div(H2(todo['title']), Div(todo.get('details', ''), cls="markdown"), btn)

serve()
