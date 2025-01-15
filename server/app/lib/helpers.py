import os
import json
import pickle
from functools import wraps

def get_model_dims(model_kind, dense_model_name):
    mods = model_kind.list_supported_models()
    matching_models = [item for item in mods if item.get('model') == dense_model_name]
    if len(matching_models) != 1:
        raise ValueError(f"No matching dimensions found for '{dense_model_name}'")
    matching_model = matching_models[0]
    return matching_model.get('dim')

def cache_to_file(cache_file):
    """
    Decorator to cache a function's output to a pickle file on disk.
    Note that there is no checking of conflicting namespaces. 
    So make sure that all caches are uniquely named!
    
    Args:
        cache_file (str): Path to the cache file.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if the cache file exists
            if os.path.exists(cache_file):
                info(f"Loading cached result from {cache_file}...")
                with open(cache_file, "rb") as file:
                    return pickle.load(file)
            
            # Compute the result and save it to the cache
            result = func(*args, **kwargs)
            with open(cache_file, "wb") as file:
                pickle.dump(result, file)
            info(f"Result cached to {cache_file}.")
            return result
        
        return wrapper
    return decorator

@cache_to_file("user_payloads.pkl")
def load_data_with_assigned_users(no_of_users: int):
    """
    The data is arXiv paper titles and abstracts, available at:
    https://www.kaggle.com/datasets/Cornell-University/arxiv

    For this example, we randomly assign each paper to one of `no_of_users` different users.
    We will represent each point in Qdrant through a combined title/abstract string.
    """
    import random

    with open("./data.json") as f:
        raw_data = json.load(f)
    info("ArXiv data loaded from disk.")

    raw_data = raw_data[:1000000] 

    user_domain = range(1, no_of_users)
    user_assignments = [random.choice(user_domain) for _ in raw_data]
    data = [{
        'title': vl.get('title').replace('\\r\\n', ' ').strip(), 
        'abstract': vl.get('abstract').replace('\\r\\n', ' ').strip(),
        'user_id': user_assignments[idx],
    } for idx, vl in enumerate(raw_data)]
    info("Users assigned to all data.")

    texts: List[str] = [
        "This paper is titled '" + vl.get('title').replace('\\r\\n', ' ').strip() + "'. " + vl.get('abstract').replace('\\r\\n', ' ').strip()
        for vl in raw_data
    ]

    return (data, texts)


