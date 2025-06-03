from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Qdrant credentials from environment variables
api_key = os.getenv("QDRANT_API_KEY")
cluster_url = os.getenv("QDRANT_CLUSTER_URL")

# Connect to Qdrant cluster if credentials are available, otherwise use local
if api_key and cluster_url:
    client = QdrantClient(url=cluster_url, api_key=api_key)
    print("Connected to Qdrant cloud cluster")
else:
    client = QdrantClient(url="http://localhost:6333")
    print("Connected to local Qdrant instance")

colName = "OmiyDB"

if not client.collection_exists(collection_name=colName):
    client.create_collection(
        collection_name=colName,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE) # assumes dimensions of microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
    )
    print(f"Collection '{colName}' created successfully!")
else:
    print(f"Collection '{colName}' already exists.")