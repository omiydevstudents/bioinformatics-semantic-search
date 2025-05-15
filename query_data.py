from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient(url="http://localhost:6333")

colName = "OmiyDB"

model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
model = SentenceTransformer(model_name)
print(f"Model '{model_name}' loaded successfully!")

# test a simple search
query = "Which tool provides good workflows for germline short variant discovery from high-throughput sequencing data?"
query_vector = model.encode(query).tolist()

search_results = client.search(
    collection_name=colName,
    query_vector=query_vector,
    limit=3
)

print("\nSearch results for query:", query)
for result in search_results:
    print(f"Tool: {result.payload['tool_name']}")
    print(f"Score: {result.score}")
    print(f"Description: {result.payload['description']}")
    print(f"URL: {result.payload['url']}")
    print("-------------")