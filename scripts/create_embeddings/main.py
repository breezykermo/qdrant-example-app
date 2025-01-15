"""
This example is adapted from:
https://qdrant.github.io/fastembed/examples/Hybrid_Search

It creates a new collection for hybrid search in a Qdrant cluster from the following dataset:

https://www.kaggle.com/datasets/Cornell-University/arxiv 

Both sparse and dense embeddings are generated from the title and abstract of each paper.
All points are then uploaded to the newly created collection.  
Note that generating the embeddings can take some time (minutes), so be patient!
The weights for the model will also be downloaded over the network, so ensure you have a good connection.
"""
import os
import json
from qdrant_client import QdrantClient
from qdrant_client.models import (
    BinaryQuantization, 
    BinaryQuantizationConfig,
    Distance,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from fastembed import (
    LateInteractionTextEmbedding,
    SparseTextEmbedding,
    TextEmbedding,
)
from yaspin import yaspin
from typing import List, Union

DEBUG = True
EMBEDDING_BATCH_SIZE = 32 

qdrant_host = os.getenv('QDRANT_HOST')
qdrant_port = os.getenv('QDRANT_PORT')
collection_name = os.getenv('QDRANT_COLLECTION_NAME')
replication_factor = os.getenv('QDRANT_REPLICATION_FACTOR')
shard_number = os.getenv('QDRANT_SHARD_NUMBER')

sparse_model_name = os.getenv('SPARSE_MODEL_NAME')
dense_model_name = os.getenv('DENSE_MODEL_NAME')
late_interaction_model_name = os.getenv('LATE_INTERACTION_MODEL_NAME')

upsert_index_start = int(os.getenv("DATASET_INDEX_START"))
upsert_index_end = int(os.getenv("DATASET_INDEX_END"))
upsert_index_start_zfill = str(upsert_index_start).zfill(8)
upsert_index_end_zfill = str(upsert_index_end).zfill(8)
should_upsert_points = int(os.getenv("SHOULD_UPSERT_POINTS"))

def info(*args): print(*args) if DEBUG else None 

def cache_to_file(cache_file):
    """
    Decorator to cache a function's output to a pickle file on disk.
    Note that there is no checking of conflicting namespaces. 
    So make sure that all caches are uniquely named!
    
    Args:
        cache_file (str): Path to the cache file.
    """
    import pickle
    from functools import wraps
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

def make_embeddings(model, docs: List[str]):
    return list(
        model.embed(docs, batch_size=EMBEDDING_BATCH_SIZE)
    )

# Initialize Qdrant client
client = QdrantClient(host=qdrant_host, port=qdrant_port)
info("Qdrant client configured.")

@yaspin(text="Loading data from disk...")
@cache_to_file(f"data/{upsert_index_start_zfill}_{upsert_index_end_zfill}_payloads.pkl")
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

    # Take start and end indices from environment 
    raw_data = raw_data[upsert_index_start:upsert_index_end] 

    user_domain = range(1, no_of_users)
    user_assignments = [random.choice(user_domain) for _ in raw_data]
    data = [{
        'title': vl.get('title').replace('\\r\\n', ' ').strip(), 
        'abstract': vl.get('abstract').replace('\\r\\n', ' ').strip(),
        'user_id': user_assignments[idx],
    } for idx, vl in enumerate(raw_data)]
    info("Users assigned to all data.")

    # texts: List[str] = [
    #     "This paper is titled '" + vl.get('title').replace('\\r\\n', ' ').strip() + "'. " + vl.get('abstract').replace('\\r\\n', ' ').strip()
    #     for vl in raw_data
    # ]
    texts: List[str] = [vl.get('title') for vl in raw_data]

    return (data, texts)

@yaspin(text="Generating sparse embeddings...")
@cache_to_file(f"data/{upsert_index_start_zfill}_{upsert_index_end_zfill}_sparse.pkl")
def make_sparse_embeddings(texts: List[str]):
    sparse_model = SparseTextEmbedding(model_name=sparse_model_name, batch_size=EMBEDDING_BATCH_SIZE)
    sparse_embeddings = make_embeddings(sparse_model, texts) 
    info("Sparse embeddings generated for all texts")

    return sparse_embeddings

@yaspin(text="Generating dense embeddings...")
@cache_to_file(f"data/{upsert_index_start_zfill}_{upsert_index_end_zfill}_dense.pkl")
def make_dense_embeddings(texts: List[str]):
    dense_model = TextEmbedding(model_name=dense_model_name, batch_size=EMBEDDING_BATCH_SIZE)
    dense_embeddings = make_embeddings(dense_model, texts) 
    info("Dense embeddings generated for all texts")

    return dense_embeddings

@yaspin(text="Generating late interaction embeddings...", timer=True)
@cache_to_file(f"data/{upsert_index_start_zfill}_{upsert_index_end_zfill}_lateinteraction.pkl")
def make_late_interaction_embeddings(texts: List[str]):
    late_interaction_model = LateInteractionTextEmbedding(model_name=late_interaction_model_name, batch_size=EMBEDDING_BATCH_SIZE)
    late_interaction_embeddings = make_embeddings(late_interaction_model, texts) 
    info("Late interaction embeddings generated for all texts")

    return late_interaction_embeddings

@cache_to_file(f"data/points{upsert_index_start_zfill}_{upsert_index_end_zfill}.pkl")
def make_points():
    payloads, texts = load_data_with_assigned_users(10)
    sparse_embeddings = make_sparse_embeddings(texts)
    dense_embeddings = make_dense_embeddings(texts)
    late_interaction_embeddings = make_late_interaction_embeddings(texts)

    return [PointStruct(
        id=idx,
        payload={
            'title': payload.get('title'),
            'abstract': payload.get('abstract'),
            'user_id': payload.get('user_id'),
        }, 
        vector={
            "text-sparse": SparseVector(
                indices=sparse_vector.indices.tolist(),
                values=sparse_vector.values.tolist(),
            ),
            "text-dense": dense_vector.tolist(),
            "text-late-interaction": late_interaction_vector,
        },
    ) for idx, (dense_vector, sparse_vector, late_interaction_vector, payload)
        in enumerate(zip(dense_embeddings, sparse_embeddings, late_interaction_embeddings, payloads))]

points = make_points()

def chunk_list(lst, chunk_size=50):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

# NOTE: operation seems to time out with bigger upserts
if should_upsert_points != 0:
    for i, chunk in enumerate(chunk_list(points)):
        t = client.upsert(collection_name, chunk)
        info(f"Upserted chunk {i}")
