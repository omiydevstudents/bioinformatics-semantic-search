from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from tabulate import tabulate

# Load environment variables (optional)
load_dotenv()

# Define test cases with expected tools that should be retrieved
test_cases = [
    {
        "query": "I need to align multiple protein sequences together",
        "expected_tools": ["Clustal Omega", "BioPython"],
        "keywords": ["alignment", "sequences", "protein"]
    },
    {
        "query": "Which tool can help me visualize genomic data?",
        "expected_tools": ["IGV (Integrative Genomics Viewer)", "Cytoscape"],
        "keywords": ["visualize", "genomic", "data"]
    },
    {
        "query": "I'm looking for a tool to analyze RNA-seq data",
        "expected_tools": ["Bioconductor", "Galaxy", "GATK (Genome Analysis Toolkit)"],
        "keywords": ["RNA-seq", "analysis"]
    },
    {
        "query": "What's good for protein-protein interaction networks?",
        "expected_tools": ["Cytoscape"],
        "keywords": ["protein-protein", "networks", "interaction"]
    },
    {
        "query": "I need something for finding similar sequences in a database",
        "expected_tools": ["BLAST (Basic Local Alignment Search Tool)"],
        "keywords": ["similar", "sequences", "database"]
    }
]

def evaluate_retrieval_quality():
    """
    Evaluate the quality of vector retrieval without using any LLM API.
    This test helps assess if the semantic search is returning relevant tools.
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
    
    print("\n=== Evaluating Vector Retrieval Quality ===")
    print("Testing how well the semantic search matches expected tools\n")
    
    # Prepare results table
    results = []
    total_matches = 0
    total_expected = 0
    
    # Evaluate each test case
    for i, case in enumerate(test_cases, 1):
        query = case["query"]
        expected_tools = case["expected_tools"]
        keywords = case["keywords"]
        
        print(f"\nTest {i}: '{query}'")
        print(f"Keywords: {', '.join(keywords)}")
        print(f"Expected tools: {', '.join(expected_tools)}")
        
        # Perform vector search
        k = max(3, len(expected_tools))  # Retrieve at least 3 results or more if needed
        retrieved_docs = vector_store.similarity_search(query, k=k)
        retrieved_tools = [doc.metadata['tool_name'] for doc in retrieved_docs]
        
        print(f"Retrieved tools: {', '.join(retrieved_tools)}")
        
        # Calculate metrics
        matches = set(expected_tools).intersection(set(retrieved_tools))
        match_count = len(matches)
        precision = match_count / len(retrieved_tools) if retrieved_tools else 0
        recall = match_count / len(expected_tools) if expected_tools else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Update totals
        total_matches += match_count
        total_expected += len(expected_tools)
        
        # Display match results
        print(f"Matches: {', '.join(matches) if matches else 'None'}")
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
        
        # Add to results table
        results.append([
            f"Test {i}",
            query,
            ', '.join(expected_tools),
            ', '.join(retrieved_tools),
            match_count,
            f"{precision:.2f}",
            f"{recall:.2f}",
            f"{f1:.2f}"
        ])
    
    # Calculate overall metrics
    overall_precision = total_matches / (len(test_cases) * k) if test_cases else 0
    overall_recall = total_matches / total_expected if total_expected else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    # Add summary row
    results.append([
        "Overall",
        "-",
        "-",
        "-",
        total_matches,
        f"{overall_precision:.2f}",
        f"{overall_recall:.2f}",
        f"{overall_f1:.2f}"
    ])
    
    # Display summary table
    print("\n=== Evaluation Summary ===")
    print(tabulate(
        results,
        headers=["Test", "Query", "Expected Tools", "Retrieved Tools", "Matches", "Precision", "Recall", "F1 Score"],
        tablefmt="grid"
    ))
    
    # Provide assessment
    print("\n=== Retrieval Quality Assessment ===")
    if overall_f1 >= 0.7:
        print("✓ Excellent retrieval quality! Your vector search is performing very well.")
    elif overall_f1 >= 0.5:
        print("✓ Good retrieval quality. Your vector search is working reasonably well.")
    elif overall_f1 >= 0.3:
        print("△ Fair retrieval quality. Consider adding more data or tuning your embeddings.")
    else:
        print("× Poor retrieval quality. Check your embeddings and collection setup.")
    
    print("\nNote: This evaluation focuses on retrieval quality, not the quality of LLM responses.")
    print("It shows how well your vector search can find relevant tools for user queries.")

if __name__ == "__main__":
    evaluate_retrieval_quality() 