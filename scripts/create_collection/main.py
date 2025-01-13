import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

qdrant_host = os.getenv('QDRANT_HOST')
qdrant_port = os.getenv('QDRANT_PORT')
collection_name = os.getenv('QDRANT_COLLECTION_NAME')
replication_factor = os.getenv('QDRANT_REPLICATION_FACTOR')
shard_number = os.getenv('QDRANT_SHARD_NUMBER')

client = QdrantClient(host=qdrant_host, port=qdrant_port)

print("Connected to Qdrant cluster.")

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=300, distance=Distance.COSINE),
        replication_factor=replication_factor,
        shard_number=shard_number,
    )

    print(f"Collection '{collection_name}' created with {shard_number} logical shards, replicated across {replication_factor} nodes.")
