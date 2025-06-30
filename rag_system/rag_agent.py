"""
Bioinformatics RAG Agent using LangChain, LangGraph, and Qdrant
Enhanced with Self-RAG capabilities for better quality control
"""

from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

_qdrant_client = None
_llm = None
_embedding_model = None
_relevance_grader = None
_hallucination_grader = None
_answer_grader = None
_query_rewriter = None

# Pydantic models for structured output
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved bioinformatics tools."""
    binary_score: str = Field(
        description="Tool is relevant to the question, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the tool facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

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

def get_relevance_grader():
    """Get or create relevance grader (singleton pattern)"""
    global _relevance_grader
    if _relevance_grader is None:
        llm = get_llm()
        structured_llm_grader = llm.with_structured_output(GradeDocuments)
        
        system_prompt = """You are a grader assessing relevance of a bioinformatics tool to a user question.
It does not need to be a stringent test. The goal is to filter out clearly irrelevant tools.
If the tool contains functionality or purpose related to the user question, grade it as relevant.

Consider these connections:
- Sequence analysis tools are relevant to genomics/DNA/RNA questions
- Quality control tools are relevant to data processing questions  
- Visualization tools are relevant to data analysis questions
- Statistical tools are relevant to computational biology questions
- Assembly tools are relevant to genome/transcriptome construction
- Alignment tools are relevant to sequence comparison questions

Give a binary score 'yes' or 'no' to indicate whether the tool is relevant to the question."""

        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Retrieved tool:\n\nName: {tool_name}\nDescription: {tool_description}\nTopics: {tool_topics}\nOperations: {tool_operations}\n\nUser question: {question}")
        ])
        
        _relevance_grader = grade_prompt | structured_llm_grader
    
    return _relevance_grader

def get_hallucination_grader():
    """Get or create hallucination grader (singleton pattern)"""
    global _hallucination_grader
    if _hallucination_grader is None:
        llm = get_llm()
        structured_llm_grader = llm.with_structured_output(GradeHallucinations)
        
        system_prompt = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved bioinformatics tool facts.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the tool information.
Check that:
1. Tool names mentioned in the answer exist in the retrieved tools
2. Features/capabilities mentioned are actually described in the tool information
3. No made-up functionality is attributed to the tools"""

        hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Set of tool facts:\n\n{tools}\n\nLLM generation: {generation}")
        ])
        
        _hallucination_grader = hallucination_prompt | structured_llm_grader
    
    return _hallucination_grader

def get_answer_grader():
    """Get or create answer grader (singleton pattern)"""
    global _answer_grader
    if _answer_grader is None:
        llm = get_llm()
        structured_llm_grader = llm.with_structured_output(GradeAnswer)
        
        system_prompt = """You are a grader assessing whether an answer addresses / resolves a bioinformatics question.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer provides useful tools or information that would help solve the user's bioinformatics problem."""

        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "User question: {question}\n\nLLM generation: {generation}")
        ])
        
        _answer_grader = answer_prompt | structured_llm_grader
    
    return _answer_grader

def get_query_rewriter():
    """Get or create query rewriter (singleton pattern)"""
    global _query_rewriter
    if _query_rewriter is None:
        llm = get_llm()
        
        system_prompt = """You are a question re-writer that converts an input bioinformatics question to a better version that is optimized for tool discovery.
Look at the input and try to reason about the underlying bioinformatics task or workflow.

Examples:
- "analyze gene expression" → "differential gene expression analysis tools RNA-seq"
- "protein structure" → "protein structure prediction modeling visualization"
- "sequence alignment" → "multiple sequence alignment pairwise alignment tools"

Focus on the core bioinformatics operations and add relevant technical terms."""

        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Here is the initial question: {question}\nFormulate an improved question for finding bioinformatics tools.")
        ])
        
        _query_rewriter = rewrite_prompt | llm | StrOutputParser()
    
    return _query_rewriter

# Enhanced state for Self-RAG workflow
class RAGState(TypedDict):
    """Enhanced state for the Self-RAG agent workflow"""
    user_query: str
    query_embedding: List[float]
    search_results: List[Dict[str, Any]]
    formatted_answer: str
    relevance_check_passed: bool
    max_retries: int
    current_retry: int

# Define the workflow nodes (keeping existing + adding new)
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
        limit=5,  # Get more tools for better filtering
        with_payload=True,
    )
    
    # Extract the relevant information
    tools_found = []
    for hit in search_results:
        tool_info = {
            "name": hit.payload.get("name", "Unknown"),
            "description": hit.payload.get("description", "No description"),
            "homepage": hit.payload.get("homepage", "No URL"),
            "topics": hit.payload.get("topics", []),
            "operations": hit.payload.get("operations", []),
            "language": hit.payload.get("language", []),
            "biotools_id": hit.payload.get("biotools_id", ""),
            "relevance_score": hit.score
        }
        tools_found.append(tool_info)
    
    state["search_results"] = tools_found
    print(f"Found {len(tools_found)} candidate tools")
    
    return state

def grade_tool_relevance(state: RAGState) -> RAGState:
    """Grade each retrieved tool for relevance to the question"""
    print("Step 2.1: Grading tool relevance...")
    
    question = state["user_query"]
    tools = state["search_results"]
    relevance_grader = get_relevance_grader()
    
    relevant_tools = []
    
    for tool in tools:
        # Prepare grading input
        grading_input = {
            "question": question,
            "tool_name": tool['name'],
            "tool_description": tool['description'],
            "tool_topics": ', '.join(tool['topics']) if tool['topics'] else 'N/A',
            "tool_operations": ', '.join(tool['operations']) if tool['operations'] else 'N/A'
        }
        
        # Grade relevance
        grade_result = relevance_grader.invoke(grading_input)
        
        if grade_result.binary_score == "yes":
            relevant_tools.append(tool)
            print(f"✅ {tool['name']} - RELEVANT")
        else:
            print(f"❌ {tool['name']} - NOT RELEVANT")
    
    state["search_results"] = relevant_tools
    state["relevance_check_passed"] = len(relevant_tools) > 0
    print(f"Filtered to {len(relevant_tools)} relevant tools")
    
    return state

def format_answer_with_llm(state: RAGState) -> RAGState:
    """Use Gemini to format a helpful answer based on search results"""
    print("Step 3: Formatting answer with LLM...")
    
    llm = get_llm()
    
    # Create context from search results
    tools_context = "\n\n".join([
        f"Tool: {tool['name']}\n"
        f"Description: {tool['description']}\n"
        f"Topics: {', '.join(tool['topics']) if tool['topics'] else 'N/A'}\n"
        f"Operations: {', '.join(tool['operations']) if tool['operations'] else 'N/A'}\n"
        f"Language: {', '.join(tool['language']) if tool['language'] else 'N/A'}\n"
        f"URL: {tool['homepage']}\n"
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

def transform_query(state: RAGState) -> RAGState:
    """Transform the query to produce a better question for tool discovery"""
    print("Step 4: Transforming query for better results...")
    
    query_rewriter = get_query_rewriter()
    
    # Rewrite the query
    better_question = query_rewriter.invoke({"question": state["user_query"]})
    
    print(f"Original query: {state['user_query']}")
    print(f"Improved query: {better_question}")
    
    # Update query and increment retry counter
    state["user_query"] = better_question
    state["current_retry"] = state.get("current_retry", 0) + 1
    
    return state

# Decision functions
def decide_to_generate(state: RAGState) -> str:
    """Determines whether to generate an answer or transform the query"""
    print("---ASSESS GRADED TOOLS---")
    
    if not state["relevance_check_passed"]:
        if state.get("current_retry", 0) < state.get("max_retries", 2):
            print("---DECISION: NO RELEVANT TOOLS FOUND, TRANSFORM QUERY---")
            return "transform_query"
        else:
            print("---DECISION: MAX RETRIES REACHED, GENERATE WITH AVAILABLE TOOLS---")
            return "generate"
    else:
        print("---DECISION: RELEVANT TOOLS FOUND, GENERATE ANSWER---")
        return "generate"

def grade_generation_quality(state: RAGState) -> str:
    """Check if the generation is grounded and addresses the question"""
    print("Step 3.1: Checking answer quality...")
    
    question = state["user_query"]
    tools = state["search_results"]
    generation = state["formatted_answer"]
    
    # Create tools context for hallucination check
    tools_context = "\n\n".join([
        f"Tool: {tool['name']}\nDescription: {tool['description']}\nTopics: {tool['topics']}\nOperations: {tool['operations']}"
        for tool in tools
    ])
    
    # Check for hallucinations
    hallucination_grader = get_hallucination_grader()
    hallucination_score = hallucination_grader.invoke({
        "tools": tools_context,
        "generation": generation
    })
    
    if hallucination_score.binary_score == "yes":
        print("✅ GENERATION IS GROUNDED IN TOOL FACTS")
        
        # Check if answer addresses the question
        answer_grader = get_answer_grader()
        answer_score = answer_grader.invoke({
            "question": question,
            "generation": generation
        })
        
        if answer_score.binary_score == "yes":
            print("✅ GENERATION ADDRESSES THE QUESTION")
            return "useful"
        else:
            print("❌ GENERATION DOES NOT ADDRESS THE QUESTION")
            if state.get("current_retry", 0) < state.get("max_retries", 2):
                return "not_useful"
            else:
                return "useful"  # Accept answer after max retries
    else:
        print("❌ GENERATION CONTAINS HALLUCINATIONS")
        if state.get("current_retry", 0) < state.get("max_retries", 2):
            return "not_supported"
        else:
            return "useful"  # Accept answer after max retries

# Create the enhanced LangGraph workflow
def create_rag_workflow():
    """Create and compile the enhanced Self-RAG workflow graph"""
    
    # Initialize the graph
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("embed_query", embed_query)
    workflow.add_node("search_vector_db", search_vector_db)
    workflow.add_node("grade_tool_relevance", grade_tool_relevance)
    workflow.add_node("generate", format_answer_with_llm)
    workflow.add_node("transform_query", transform_query)
    
    # Define the flow
    workflow.add_edge(START, "embed_query")
    workflow.add_edge("embed_query", "search_vector_db")
    workflow.add_edge("search_vector_db", "grade_tool_relevance")
    
    # Conditional edge: decide whether to generate or transform query
    workflow.add_conditional_edges(
        "grade_tool_relevance",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        }
    )
    
    # Transform query loops back to embedding
    workflow.add_edge("transform_query", "embed_query")
    
    # Conditional edge: check generation quality
    workflow.add_conditional_edges(
        "generate",
        grade_generation_quality,
        {
            "not_supported": "generate",  # Regenerate if hallucinating
            "not_useful": "transform_query",  # Transform query if not useful
            "useful": END,  # End if answer is good
        }
    )
    
    # Compile the graph
    app = workflow.compile()
    
    return app

# Main function to run the enhanced agent
def query_bioinformatics_tools(user_query: str):
    """Main function to query bioinformatics tools with Self-RAG"""
    print(f"\n🔍 Processing query with Self-RAG: '{user_query}'\n")
    
    # Create the workflow
    rag_app = create_rag_workflow()
    
    # Run the workflow
    result = rag_app.invoke({
        "user_query": user_query,
        "query_embedding": [],
        "search_results": [],
        "formatted_answer": "",
        "relevance_check_passed": False,
        "max_retries": 2,
        "current_retry": 0
    })
    
    return result["formatted_answer"]