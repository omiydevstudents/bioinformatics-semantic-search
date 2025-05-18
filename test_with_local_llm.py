from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load environment variables (optional)
load_dotenv()

def test_with_local_llm():
    """
    Test the RAG pipeline with a local, small LLM instead of using API keys.
    This uses a free, smaller model from Hugging Face that can run locally.
    """
    print("Loading biomedical embedding model for vector search...")
    # Load the biomedical BERT model for embeddings
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
    
    # Create a retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    print("\nLoading a small local language model...")
    print("This might take a minute to download the first time...")
    
    # Load a small local LLM (TinyLlama)
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Set up tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    
    # Create a text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
    )
    
    # Create LangChain HF pipeline
    local_llm = HuggingFacePipeline(pipeline=pipe)
    
    print("✓ Local language model loaded successfully!")
    
    # Create a prompt template
    template = """You are a bioinformatics expert assistant.
    Answer the following question about bioinformatics tools based on the context provided.
    Keep your answer brief and to the point.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | local_llm
        | StrOutputParser()
    )
    
    print("\n=== Testing RAG with Local LLM ===")
    print("This will perform semantic search and then generate an answer using a local model\n")
    
    # Example query
    test_query = "Which tool should I use for protein-protein interaction analysis?"
    
    print(f"Query: '{test_query}'")
    print("Retrieving relevant information and generating response...")
    
    # Run the query
    response = rag_chain.invoke(test_query)
    
    print("\nResponse from local LLM:")
    print("-" * 40)
    print(response)
    print("-" * 40)
    
    print("\n✓ Test completed successfully!")
    print("The RAG pipeline works with a local LLM.")
    print("\nNote: This small model might not give the best quality responses.")
    print("For better results, consider using:")
    print("1. A larger local model if you have the computational resources")
    print("2. Gemini or OpenAI API with an API key")

if __name__ == "__main__":
    test_with_local_llm() 