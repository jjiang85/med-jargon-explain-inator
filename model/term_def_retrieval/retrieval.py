import json
import os

"""
Notes for me:
- Load JSON with each call
- Each call retrieves a single term or definition
- Function takes a str
"""




def get_definition(sample: str, term_to_cui_file: str = "../../data/term_to_cui.json", 
                   cui_to_def_file: str = "../../data/cui_to_def.json") -> dict:
    """
    Retrieves potential definitions for a given term or phrase.

    Currently only identifies jargon in the list based on exact match.
    
    Arguments:
        sample (str): the term or phrase to look up in the jargon list and define.

    Returns:
        dict{source1: def1, source2: def2, ...}:
            if sample appears in jargon list.
        dict{}:
            if sample is found but not defined.
        None: 
            if sample DOES NOT appear in jargon list.
    """
    sample_term = sample.lower().strip()
    
    # check if sample is in jargon list
    with open(term_to_cui_file, 'r') as f:
        term_to_cui = json.load(f)

    if sample_term in term_to_cui.keys():
        sample_cuis = term_to_cui[sample_term]   # list of all possible CUIs
    else: 
        return None   # sample not in jargon list:

    # retrieve definitions for sample
    with open(cui_to_def_file, 'r') as f:
        cui_to_def = json.load(f)

    for cui in sample_cuis:
        # retrieve empty dictionary if not defined
        definitions = cui_to_def.get(cui, {})

    return definitions