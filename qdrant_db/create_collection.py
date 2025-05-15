from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(url="http://localhost:6333")

colName = "OmiyDB"

if not client.collection_exists(collection_name=colName):
    client.create_collection(
        collection_name=colName,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE) # assumes dimensions of microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
    )
    print(f"Collection '{colName}' created successfully!")
else:
    print(f"Collection '{colName}' already exists.")