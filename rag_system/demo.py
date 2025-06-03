"""
Simple demo script for the Bioinformatics RAG system
Run this to quickly test the system with predefined queries
"""

from rag_agent import query_bioinformatics_tools
from rag_utils import validate_environment, test_connections

def run_demo():
    """Run a simple demo of the RAG system"""
    
    print("🧬 Welcome to the Bioinformatics Tool Finder!")
    print("=" * 50)
    
    print("\n📋 Checking system...")

    if validate_environment():
        print("✅ Environment variables are correctly set.")
    else:
        print("❌ Please set the required environment variables in your .env file.")
        return
    
    if test_connections():
        print("✅ Connections to Qdrant and Gemini are working.")
    else:
        print("❌ There was an error with the connections. Please check your setup.")
        return
    
    print("\n✅ System checks are ready!\n")
    
    while True:
        user_query = input("\n💬 Ask about bioinformatics tools (or 'quit' to exit): ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thanks for using the Bioinformatics Tool Finder!")
            break
        
        if not user_query:
            print("Please enter a question.")
            continue
        
        try:
            answer = query_bioinformatics_tools(user_query)
            print("\n💡 Answer:")
            print(answer)
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    run_demo()