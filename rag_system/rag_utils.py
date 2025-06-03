"""
Utility functions for the Bioinformatics RAG system
"""

import os
from dotenv import load_dotenv

load_dotenv()

def validate_environment():
    """Check if all required environment variables are set"""
    required_vars = [
        "QDRANT_API_KEY",
        "QDRANT_CLUSTER_URL", 
        "GOOGLE_API_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        return False
    
    return True

def test_connections():
    """Test connections to Qdrant and Gemini"""
    print("\n🔧 Testing Qdrant and Gemini connections...")
    
    # Test Qdrant
    try:
        from qdrant_client import QdrantClient
        api_key = os.getenv("QDRANT_API_KEY")
        cluster_url = os.getenv("QDRANT_CLUSTER_URL")
        
        if api_key and cluster_url:
            client = QdrantClient(url=cluster_url, api_key=api_key)
            collections = client.get_collections()
            print(f"✅ Qdrant connection successful. Collections: {[c.name for c in collections.collections]}")
        else:
            # throw an error if no API key or URL is provided (there is no local Qdrant instance)
            raise ValueError("There was an error connecting to Qdrant. Please check your API key and cluster URL.")
    except Exception as e:
        print(f"❌ Qdrant connection failed: {str(e)}")
        return False
    
    # Test Gemini
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        # Simple test
        response = llm.invoke("Say 'connection successful' in 3 words")
        if response.content:
            print(f"✅ Gemini connection successful: {response.content}")
        else:
            raise ValueError("There was an error connecting to Gemini. Please check your API key.")
    except Exception as e:
        print(f"❌ Gemini connection failed: {str(e)}")
        return False
    
    return True
