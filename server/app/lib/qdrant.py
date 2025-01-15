import os
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import (
    BinaryQuantization, 
    BinaryQuantizationConfig,
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    MultiVectorComparator,
    MultiVectorConfig,
    NamedSparseVector,
    PointStruct,
    Prefetch,
    SearchRequest,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from fastembed import TextEmbedding, LateInteractionTextEmbedding
from .helpers import cache_to_file, get_model_dims

# Initialize Qdrant client
qdrant_host = os.getenv('QDRANT_HOST')
qdrant_port = os.getenv('QDRANT_PORT')
client = QdrantClient(host=qdrant_host, port=qdrant_port)

def make_embeddings(model, doc: str):
    return next(model.embed([doc]))

def init_collection(logger, collection_name):
    """
    Initialize collection in Qdrant cluster.
    Text in the collection is represented by three vectors:
    - A dense vector for semantic search
    - A sparse vector for keyword information retrieval
    - A late interaction vector for effective re-ranking
    """
    shard_number = os.getenv("QDRANT_SHARD_NUMBER")
    replication_factor = os.getenv("QDRANT_REPLICATION_FACTOR")

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "text-dense": VectorParams(
                    size=get_model_dims(TextEmbedding, os.getenv("DENSE_MODEL_NAME")),
                    distance=Distance.COSINE
                ),
                "text-late-interaction": VectorParams(
                    size=get_model_dims(LateInteractionTextEmbedding, os.getenv("LATE_INTERACTION_MODEL_NAME")),
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM,
                    )
                ),
            },
            sparse_vectors_config={
                "text-sparse": SparseVectorParams(),
            },
            # Enable binary quantization for collection
            quantization_config=BinaryQuantization(
                binary=BinaryQuantizationConfig(
                    always_ram=False,
                ),
            ),
            replication_factor=replication_factor,
            shard_number=shard_number,
        )

        logger.info(f"Collection '{collection_name}' created with {shard_number} logical shards, replicated across {replication_factor} nodes.")
    else:
        logger.info(f"Collection '{collection_name}' already exists, nothing done.")

def search(user_id, collection_name, sparse_model, dense_model, late_interaction_model, query_text: str):
    """
    Use a hybrid search (using both sparse vectors and dense vectors) that is re-ranked with a late interaction model.
    Searches are filtered by "user_id"
    """ 
    query_sparse_vectors  = make_embeddings(sparse_model, query_text)
    query_dense_vectors = make_embeddings(dense_model, query_text)
    query_late_interaction_vectors = make_embeddings(late_interaction_model, query_text)

    prefetch = [
        Prefetch(
            query=query_dense_vectors,
            using='text-dense',
            limit=20,
        ),
        Prefetch(
            query=SparseVector(**query_sparse_vectors.as_object()),
            using='text-sparse',
            limit=20,
        ),
    ]

    filter = Filter(
        must=[
            FieldCondition(
                key='user_id',
                match=MatchValue(value=user_id)
            ),
        ],
    )

    search_results = client.query_points(
        collection_name=collection_name,
        prefetch=prefetch,
        query_filter=filter,
        query=query_late_interaction_vectors,
        using='text-late-interaction',
        with_payload=True,
        limit=10,
    )


    return search_results


