import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()

def create_bioinformatics_assistant():
    """
    Creates a bioinformatics assistant using Gemini that can answer questions about bioinformatics tools
    and maintain conversation context.
    """
    # Load the same biomedical BERT model used in your existing code
    embeddings = HuggingFaceEmbeddings(
        model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("Embeddings model loaded successfully!")
    
    # Connect to the existing Qdrant collection
    vector_store = Qdrant(
        client_url="http://localhost:6333",
        collection_name="OmiyDB",  # Use your existing collection
        embeddings=embeddings,
    )
    
    # Create a retriever from the vector store
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Create conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create the conversational retrieval chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )
    
    return qa_chain

def main():
    # Check if Google API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY environment variable is not set.")
        print("Please set it in a .env file or directly in your environment.")
        print("For example: GOOGLE_API_KEY=your-api-key-here")
        return
    
    print("🧬 Bioinformatics Assistant powered by Gemini 🧬")
    print("Ask me about bioinformatics tools and methods!")
    print("Type 'exit' to quit.")
    
    # Create the assistant
    assistant = create_bioinformatics_assistant()
    
    # Interactive loop
    while True:
        user_input = input("\nYour question: ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye! Happy researching!")
            break
        
        # Get response from the assistant
        response = assistant({"question": user_input})
        
        # Print the response
        print("\nBioinformatics Assistant:", response["answer"])

if __name__ == "__main__":
    main() 