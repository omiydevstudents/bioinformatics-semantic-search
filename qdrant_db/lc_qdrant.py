from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Initialize the Qdrant client
client = QdrantClient(url="http://localhost:6333")
collection_name = "OmiyDB"

# First, ensure the model is downloaded
model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
model = SentenceTransformer(model_name)
print(f"Model '{model_name}' loaded successfully!")

# Initialize the embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name=model_name
)

# Create the LangChain vector store with correct payload keys
vector_store = Qdrant(
    client=client,
    collection_name=collection_name,
    embeddings=embeddings,
    content_payload_key="description",  # The description is the main content
    metadata_payload_key=None  # The entire payload is metadata
)

# Test the vector store with a query
def test_vector_store():
    query = "Which tool provides good workflows for germline short variant discovery from high-throughput sequencing data?"
    
    # Perform similarity search
    docs = vector_store.similarity_search(
        query=query,
        k=3  # Return top 3 results
    )
    
    # Print results
    print("\nSearch results for query:", query)
    for doc in docs:
        print("\nTool:", doc.metadata.get('tool_name', 'N/A'))
        print("Description:", doc.page_content)
        print("URL:", doc.metadata.get('url', 'N/A'))
        print("-------------")

if __name__ == "__main__":
    test_vector_store() 