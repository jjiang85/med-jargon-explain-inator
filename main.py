from fastapi import FastAPI
from model import controller

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/explain/{term}")
def explain(term: str):
    return controller.explain(term)

@app.get("/define/{term}")
def define(term: str):
    term_file = "data/term_to_cui.json"
    def_file = "data/cui_to_def.json"
    return controller.define(term, term_file, def_file)