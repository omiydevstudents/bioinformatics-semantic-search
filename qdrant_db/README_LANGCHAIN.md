# LangChain Vector Store Implementation

This module implements a semantic search system for bioinformatics tools using LangChain and Qdrant. It provides a modern interface for searching through bioinformatics tools using natural language queries.

## Overview

The `lc_qdrant.py` file creates a bridge between your existing Qdrant database and LangChain's powerful retrieval capabilities. It uses the BiomedBERT model for generating embeddings and provides a clean interface for semantic search.

## Features

- **Semantic Search**: Find bioinformatics tools based on natural language queries
- **BiomedBERT Integration**: Uses domain-specific embeddings for better understanding of bioinformatics terminology
- **Structured Results**: Returns tool information in a consistent format
- **LangChain Integration**: Ready for advanced features like RAG and conversational interfaces

## Prerequisites

- Python 3.8+
- Qdrant server running
- Required packages:
  ```bash
  pip install -U langchain-huggingface langchain-qdrant sentence-transformers
  ```

## Usage

1. **Start Qdrant Server**:
   ```bash
   docker run -d -p 6333:6333 -p 6334:6334 -v ${PWD}/qdrant_storage:/qdrant/storage --name qdrant-bioinformatics qdrant/qdrant
   ```

2. **Run the Vector Store**:
   ```bash
   python lc_qdrant.py
   ```

## Code Structure

```python
# Initialize components
client = QdrantClient(url="http://localhost:6333")
embeddings = HuggingFaceEmbeddings(model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
vector_store = Qdrant(client=client, collection_name="OmiyDB", embeddings=embeddings)

# Perform search
docs = vector_store.similarity_search(query="your question here", k=3)
```

## Output Format

The search returns documents with the following structure:
- `page_content`: The tool's description
- `metadata`: Dictionary containing:
  - `tool_name`: Name of the bioinformatics tool
  - `url`: Tool's website URL
  - Other metadata fields

## Integration with LangChain

This implementation serves as the foundation for:
1. **RAG (Retrieval Augmented Generation)**
   - Combine with LLMs for natural language responses
   - Generate detailed explanations about tools

2. **Conversational Interfaces**
   - Maintain context across multiple queries
   - Provide follow-up suggestions

3. **Advanced Search Features**
   - MMR (Maximum Marginal Relevance) for diverse results
   - Metadata filtering
   - Hybrid search capabilities

## Next Steps

1. **Add LLM Integration**
   - Implement Gemini or other LLM for natural language responses
   - Create a conversational interface

2. **Enhance Retrieval**
   - Add MMR search for diverse results
   - Implement metadata filtering
   - Add hybrid search capabilities

3. **Build User Interface**
   - Create a web interface
   - Add interactive features

## Troubleshooting

1. **Model Loading Issues**
   - Ensure you have enough disk space for the BiomedBERT model
   - Check internet connection for model download

2. **Qdrant Connection Issues**
   - Verify Qdrant server is running
   - Check port availability (6333, 6334)

3. **Search Results Issues**
   - Verify data format in Qdrant collection
   - Check content_payload_key and metadata_payload_key match your data structure

## Contributing

Feel free to:
- Add more search methods
- Implement additional features
- Improve documentation
- Report issues

## License

[Your License Here] 