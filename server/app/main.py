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
from fastapi import FastAPI, Body
from fastembed import (
    LateInteractionTextEmbedding,
    SparseTextEmbedding,
    TextEmbedding,
)
import logging 
from lib.qdrant import init_collection, search


sparse_model_name = os.getenv('SPARSE_MODEL_NAME')
dense_model_name = os.getenv('DENSE_MODEL_NAME')
late_interaction_model_name = os.getenv('LATE_INTERACTION_MODEL_NAME')
collection_name = os.getenv('QDRANT_COLLECTION_NAME')

sparse_model = None
dense_model = None
late_interaction_model = None

embedding_batch_size = 32 

def lifespan(app):
    """
    On startup, load all models into memory.
    Models will be downloaded on first startup over the network.
    Models are cached between server restarts via the 'FASTEMBED_CACHE_PATH' env var.

    """
    global sparse_model
    global dense_model
    global late_interaction_model

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("startup")

    sparse_model = SparseTextEmbedding(model_name=sparse_model_name, batch_size=embedding_batch_size) 
    logger.info("Sparse model loaded.")
    dense_model = TextEmbedding(model_name=dense_model_name, batch_size=embedding_batch_size)
    logger.info("Dense model loaded.")
    late_interaction_model = LateInteractionTextEmbedding(model_name=late_interaction_model_name, batch_size=embedding_batch_size)
    logger.info("Late interaction model loaded.")

    init_collection(logger, collection_name)

    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
def info():
    return {"Hello": "World"}

@app.post("/hybrid_search")
async def hybrid_search(payload: dict = Body(...)):
    user_id = payload.get("user_id")
    query_text = payload.get("query")

    results = search(user_id, collection_name, sparse_model, dense_model, late_interaction_model, query_text)

