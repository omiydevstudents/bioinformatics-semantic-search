"""
Bioinformatics RAG Agent using LangChain, LangGraph, and Qdrant
This is the main file that orchestrates the entire RAG workflow
"""

from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Module-level global variables (initialized once)
_qdrant_client = None
_llm = None
_embedding_model = None

# Initialize components
def get_qdrant_client():
    """Get or create Qdrant client (singleton pattern)"""
    global _qdrant_client
    if _qdrant_client is None:
        api_key = os.getenv("QDRANT_API_KEY")
        cluster_url = os.getenv("QDRANT_CLUSTER_URL")
        
        if api_key and cluster_url:
            _qdrant_client = QdrantClient(url=cluster_url, api_key=api_key)
            print("Connected to Qdrant cloud cluster")
        else:
            _qdrant_client = QdrantClient(url="http://localhost:6333")
            print("Connected to local Qdrant instance")
    
    return _qdrant_client

def get_llm():
    """Get or create LLM (singleton pattern)"""
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,
            max_tokens=1024,
        )
    return _llm

def get_embedding_model():
    """Get or create embedding model (singleton pattern)"""
    global _embedding_model
    if _embedding_model is None:
        model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model

# Define the state for our agent
class RAGState(TypedDict):
    """State for the RAG agent workflow"""
    user_query: str
    query_embedding: List[float]
    search_results: List[Dict[str, Any]]
    formatted_answer: str

# Define the workflow nodes
def embed_query(state: RAGState) -> RAGState:
    """Convert user query to vector embedding"""
    print("Step 1: Creating embedding for user query...")
    
    embedding_model = get_embedding_model()
    query_vector = embedding_model.encode(state["user_query"]).tolist()
    state["query_embedding"] = query_vector
    
    return state

def search_vector_db(state: RAGState) -> RAGState:
    """Search Qdrant for relevant bioinformatics tools"""
    print("Step 2: Searching vector database...")
    
    client = get_qdrant_client()
    
    # Search the database
    search_results = client.search(
        collection_name="OmiyDB",
        query_vector=state["query_embedding"],
        limit=3,  # Get top 3 most relevant tools
        with_payload=True,
    )
    
    # Extract the relevant information
    tools_found = []
    for hit in search_results:
        tool_info = {
            "tool_name": hit.payload.get("tool_name", "Unknown"),
            "description": hit.payload.get("description", "No description"),
            "url": hit.payload.get("url", "No URL"),
            "relevance_score": hit.score
        }
        tools_found.append(tool_info)
    
    state["search_results"] = tools_found
    print(f"Found {len(tools_found)} relevant tools")
    
    return state

def format_answer_with_llm(state: RAGState) -> RAGState:
    """Use Gemini to format a helpful answer based on search results"""
    print("Step 3: Formatting answer with LLM...")
    
    llm = get_llm()
    
    # Create context from search results
    tools_context = "\n\n".join([
        f"Tool: {tool['tool_name']}\n"
        f"Description: {tool['description']}\n"
        f"URL: {tool['url']}\n"
        f"Relevance Score: {tool['relevance_score']:.2f}"
        for tool in state["search_results"]
    ])
    
    # Create the prompt
    prompt = PromptTemplate(
        input_variables=["query", "tools_context"],
        template="""You are a bioinformatics expert assistant. A user has asked about bioinformatics tools.
        
User Query: {query}

Based on my search, here are the most relevant tools from our database:

{tools_context}

Please provide a helpful, concise answer that:
1. Directly addresses the user's query
2. Recommends the most relevant tool(s) from the search results
3. Briefly explains why each recommended tool is suitable
4. Mentions the tool's key features that relate to the user's needs
5. Provides the URL for easy access

Keep your response friendly, informative, and focused on the user's specific needs."""
    )
    
    # Create and run the chain
    chain = prompt | llm | StrOutputParser()
    
    formatted_answer = chain.invoke({
        "query": state["user_query"],
        "tools_context": tools_context
    })
    
    state["formatted_answer"] = formatted_answer
    
    return state

# Create the LangGraph workflow
def create_rag_workflow():
    """Create and compile the RAG workflow graph"""
    
    # Initialize the graph
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("embed_query", embed_query)
    workflow.add_node("search_vector_db", search_vector_db)
    workflow.add_node("format_answer", format_answer_with_llm)
    
    # Define the flow
    workflow.add_edge(START, "embed_query")
    workflow.add_edge("embed_query", "search_vector_db")
    workflow.add_edge("search_vector_db", "format_answer")
    workflow.add_edge("format_answer", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app

# Main function to run the agent
def query_bioinformatics_tools(user_query: str):
    """Main function to query bioinformatics tools"""
    print(f"\n🔍 Processing query: '{user_query}'\n")
    
    # Create the workflow
    rag_app = create_rag_workflow()
    
    # Run the workflow
    result = rag_app.invoke({
        "user_query": user_query,
        "query_embedding": [],
        "search_results": [],
        "formatted_answer": ""
    })
    
    return result["formatted_answer"]
