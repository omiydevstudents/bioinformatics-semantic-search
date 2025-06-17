from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Get Qdrant credentials from environment variables
    api_key = os.getenv("QDRANT_API_KEY")
    cluster_url = os.getenv("QDRANT_CLUSTER_URL")
    embedding_model = os.getenv("EMBEDDING_MODEL")
    collection_name = os.getenv("COLLECTION_NAME")

    if api_key and cluster_url:
        client = QdrantClient(url=cluster_url, api_key=api_key)
        print("Connected to Qdrant cloud cluster")
    else:
        client = QdrantClient(url="http://localhost:6333")
        print("Connected to local Qdrant instance")

    # Load the embedding model
    model = SentenceTransformer(embedding_model)

    # Query the database
    query = input("Enter a query: ")
    query_vector = model.encode(query).tolist()
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=3,
        with_payload=True,
        with_vectors=True
    )
    # Display the search results
    print("Search Results:")
    for hit in search_result:
        print(f"ID: {hit.id} \nScore: {hit.score} \nTool: {hit.payload['name']} \nDescription: {hit.payload['description']}\n")
        print("Vector:", hit.vector[:5], "...")
        print("-" * 40)
    # Handle case where no results are found
    if not search_result:
        print("No results found for the query.")
    


if __name__ == "__main__":
    main()