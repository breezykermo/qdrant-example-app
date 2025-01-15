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
