from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

# Load environment variables (optional)
load_dotenv()

def test_vector_search():
    """
    Test the basic vector search functionality without using any LLM API.
    This only requires the local embedding model and Qdrant.
    """
    print("Loading biomedical embedding model...")
    # Load the biomedical BERT model
    embeddings = HuggingFaceEmbeddings(
        model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✓ Embedding model loaded successfully!")
    
    print("\nConnecting to Qdrant database...")
    # Connect to the Qdrant collection
    try:
        vector_store = Qdrant(
            client_url="http://localhost:6333",
            collection_name="OmiyDB",
            embeddings=embeddings,
        )
        print("✓ Connected to Qdrant successfully!")
    except Exception as e:
        print(f"× Failed to connect to Qdrant: {e}")
        print("\nMake sure Qdrant is running and you've created the collection.")
        print("You can run these commands first:")
        print("  1. Start Qdrant: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
        print("  2. Create collection: python qdrant_db/create_collection.py")
        print("  3. Upload data: python qdrant_db/upload_data.py")
        return
    
    # Example queries to test
    test_queries = [
        "Which tool is best for analyzing protein-protein interaction networks?",
        "I need a tool for sequence alignment",
        "What can I use for genomic data visualization?",
        "Tool for R-based gene expression analysis"
    ]
    
    print("\n=== Testing Vector Search ===")
    print("This will perform semantic search without using any LLM API\n")
    
    # Test each query
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: '{query}'")
        # Perform vector search
        results = vector_store.similarity_search(query, k=2)
        
        # Display results
        print(f"Top {len(results)} results:")
        for j, doc in enumerate(results, 1):
            print(f"  Result {j}: {doc.metadata['tool_name']}")
            print(f"  URL: {doc.metadata['url']}")
            print(f"  Description snippet: {doc.page_content[:150]}...")
            print()
    
    print("\n✓ Vector search test completed successfully!")
    print("The core vector search functionality is working properly.")
    print("\nNext steps:")
    print("1. If you want to use LLMs, get API keys for Gemini or OpenAI")
    print("2. Add them to your .env file")
    print("3. Run the full bioinfo_assistant_gemini.py")

if __name__ == "__main__":
    test_vector_search() 