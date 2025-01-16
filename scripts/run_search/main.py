"""
Before you run this script, ensure that you have:
1. Started up the backend (Qdrant cluster plus server), and that the logs on the server say 'Application startup complete'.
2. You have inserted some points into the backend via the 'create_embeddings' script.

This script demos an example hybrid search in Qdrant via an API call.
Qdrant itself is abstracted through the server, so all we need is to send a POST request to it.
To ensure it is on the right network and has the right env variables, run:

```
just example
```
"""
import os
import json
import requests

server_host = os.getenv('SERVER_HOST')
server_port = os.getenv('SERVER_PORT')

print(f"Querying server on port {server_port} and host '{server_host}'...")

url = f"http://{server_host}:{server_port}/hybrid_search"
payload = {
    "user_id": 1,
    "query": "HNSW vector similarity search as applied to matters in biology"
}

try:
    response = requests.post(url, json=payload)

    # Print the response content and status code
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.json() if response.headers.get('Content-Type') == 'application/json' else response.text}")

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
