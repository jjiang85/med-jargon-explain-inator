import nltk
from model.term_def_retrieval.retrieval import get_definition

def explain(term: str) -> dict:
    return {"term": term}

def define(term: str, term_file: str, def_file: str) -> dict:
    return get_definition(term, term_file, def_file)