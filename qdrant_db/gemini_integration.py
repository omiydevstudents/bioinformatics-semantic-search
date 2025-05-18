from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Load the same biomedical BERT model used in your existing code
embeddings = HuggingFaceEmbeddings(
    model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Connect to the existing Qdrant collection
vector_store = Qdrant(
    client_url="http://localhost:6333",
    collection_name="OmiyDB",
    embeddings=embeddings,
)

# Create a retriever from the vector store
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Example: Simple retrieval
def simple_retrieval(query):
    """Perform simple retrieval from Qdrant using LangChain."""
    results = retriever.invoke(query)
    for doc in results:
        print(f"Tool: {doc.metadata['tool_name']}")
        print(f"Description: {doc.page_content}")
        print(f"URL: {doc.metadata['url']}")
        print("-------------")
    return results

# Example: RAG (Retrieval Augmented Generation) pipeline with Gemini
def create_gemini_rag_chain():
    """Create a RAG chain that combines retrieval with Gemini for better responses."""
    # Initialize Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Create a prompt template
    template = """You are a bioinformatics expert assistant.
    Answer the following question about bioinformatics tools based on the context provided.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# Example usage of the RAG chain with Gemini
def query_with_gemini_rag(query):
    """Query using the Gemini RAG pipeline for more sophisticated responses."""
    # Check if Google API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY environment variable is not set.")
        print("Please set it in your .env file or directly in your environment.")
        return None
        
    rag_chain = create_gemini_rag_chain()
    response = rag_chain.invoke(query)
    print(response)
    return response

# Example queries
if __name__ == "__main__":
    # Check for Google API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY environment variable is not set.")
        print("Please set it in your .env file or directly in your environment.")
        print("For example: GOOGLE_API_KEY=your-api-key-here")
        exit(1)
        
    test_query = "Which tool is best for analyzing protein-protein interaction networks?"
    print("Simple retrieval results:")
    simple_retrieval(test_query)
    
    print("\nGemini RAG-enhanced response:")
    query_with_gemini_rag(test_query) 