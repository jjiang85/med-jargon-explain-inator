import nltk
from model.term_def_retrieval.retrieval import get_definition

def explain(term: str) -> dict:
    return {"term": term}

def define(term: str) -> dict:
    return get_definition(term)