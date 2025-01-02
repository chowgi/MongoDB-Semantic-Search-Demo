
from fasthtml.common import *
from hmac import compare_digest
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from bson import ObjectId

# Load environment variables
load_dotenv()

# Connect to MongoDB
client = MongoClient(os.environ['MONGODB_URI'])
db = client.todos_db
todos_collection = db.todos
users_collection = db.users

@dataclass
class Todo:
    id: str
    title: str
    done: bool
    name: str
    details: str
    priority: int

@dataclass
class User:
    name: str
    pwd: str

login_redir = RedirectResponse('/login', status_code=303)

def before(req, sess):
    auth = req.scope['auth'] = sess.get('auth', None)
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
                     Script(markdown_js, type='module'))
                )
rt = app.route

@rt("/login")
def get():
    frm = Form(
        Input(id='name', placeholder='Name'),
        Input(id='pwd', type='password', placeholder='Password'),
        Button('login'),
        action='/login', method='post')
    return Titled("Login", frm)

@dataclass
class Login: name:str; pwd:str

@rt("/login")
def post(login:Login, sess):
    if not login.name or not login.pwd: return login_redir
    user = users_collection.find_one({"name": login.name})
    if not user:
        users_collection.insert_one({"name": login.name, "pwd": login.pwd})
        user = {"name": login.name, "pwd": login.pwd}
    if not compare_digest(user['pwd'].encode("utf-8"), login.pwd.encode("utf-8")): return login_redir
    sess['auth'] = user['name']
    return RedirectResponse('/', status_code=303)

@app.get("/logout")
def logout(sess):
    del sess['auth']
    return login_redir

@rt("/{fname:path}.{ext:static}")
def get(fname:str, ext:str): return FileResponse(f'{fname}.{ext}')

@patch
def __ft__(self:Todo):
    show = AX(self.title, f'/todos/{self.id}', 'current-todo')
    edit = AX('edit',     f'/edit/{self.id}' , 'current-todo')
    dt = '✅ ' if self.done else ''
    cts = (dt, show, ' | ', edit, Hidden(id="id", value=self.id), Hidden(id="priority", value=self.priority))
    return Li(*cts, id=f'todo-{self.id}')

@rt("/")
def get(auth):
    title = f"{auth}'s Todo list"
    top = Grid(H1(title), Div(A('logout', href='/logout'), style='text-align: right'))
    new_inp = Input(id="new-title", name="title", placeholder="New Todo")
    add = Form(Group(new_inp, Button("Add")),
               hx_post="/", target_id='todo-list', hx_swap="afterbegin")
    todos = [Todo(str(t['_id']), t['title'], t['done'], t['name'], t.get('details', ''), t.get('priority', 0))
            for t in todos_collection.find({"name": auth}).sort("priority")]
    frm = Form(*todos, id='todo-list', cls='sortable', hx_post="/reorder", hx_trigger="end")
    card = Card(Ul(frm), header=add, footer=Div(id='current-todo'))
    return Title(title), Container(top, card)

@rt("/reorder")
def post(id:list[str]):
    for i,id_ in enumerate(id):
        todos_collection.update_one({"_id": ObjectId(id_)}, {"$set": {"priority": i}})
    auth = request.scope['auth']
    todos = [Todo(str(t['_id']), t['title'], t['done'], t['name'], t.get('details', ''), t.get('priority', 0))
            for t in todos_collection.find({"name": auth}).sort("priority")]
    return tuple(todos)

def clr_details(): return Div(hx_swap_oob='innerHTML', id='current-todo')

@rt("/todos/{id}")
def delete(id:str):
    todos_collection.delete_one({"_id": ObjectId(id)})
    return clr_details()

@rt("/edit/{id}")
def get(id:str):
    todo = todos_collection.find_one({"_id": ObjectId(id)})
    todo_obj = Todo(str(todo['_id']), todo['title'], todo['done'], todo['name'], todo.get('details', ''), todo.get('priority', 0))
    res = Form(Group(Input(id="title"), Button("Save")),
        Hidden(id="id"), CheckboxX(id="done", label='Done'),
        Textarea(id="details", name="details", rows=10),
        hx_put="/", target_id=f'todo-{id}', id="edit")
    return fill_form(res, todo_obj)

@rt("/")
def put(todo: Todo):
    todos_collection.update_one(
        {"_id": ObjectId(todo.id)},
        {"$set": {
            "title": todo.title,
            "done": todo.done,
            "details": todo.details
        }}
    )
    updated = todos_collection.find_one({"_id": ObjectId(todo.id)})
    return Todo(str(updated['_id']), updated['title'], updated['done'], updated['name'], updated.get('details', ''), updated.get('priority', 0)), clr_details()

@rt("/")
def post(todo:Todo):
    new_inp = Input(id="new-title", name="title", placeholder="New Todo", hx_swap_oob='true')
    auth = request.scope['auth']
    result = todos_collection.insert_one({
        "title": todo.title,
        "done": False,
        "name": auth,
        "details": "",
        "priority": 0
    })
    inserted = todos_collection.find_one({"_id": result.inserted_id})
    return Todo(str(inserted['_id']), inserted['title'], inserted['done'], inserted['name'], inserted.get('details', ''), inserted.get('priority', 0)), new_inp

@rt("/todos/{id}")
def get(id:str):
    todo = todos_collection.find_one({"_id": ObjectId(id)})
    btn = Button('delete', hx_delete=f'/todos/{id}',
                 target_id=f'todo-{id}', hx_swap="outerHTML")
    return Div(H2(todo['title']), Div(todo.get('details', ''), cls="markdown"), btn)

serve()
