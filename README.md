# Bioinformatics Semantic Search

A semantic search system for bioinformatics tools using vector embeddings, designed to help researchers quickly find the right tool for their needs.

## Project Overview

This project uses the Qdrant vector database and biomedical language models to create a semantic search system for bioinformatics tools. It enables users to find relevant tools by describing their research needs in natural language, rather than relying on exact keyword matches.

## Features

- **Vector-based Semantic Search**: Find tools based on meaning, not just keywords
- **Domain-specific Embeddings**: Uses biomedical BERT model for better understanding of bioinformatics terminology
- **LangChain Integration**: Enhanced capabilities with the LangChain framework
- **Conversational Interface**: Ask questions about bioinformatics tools in natural language

## Setup and Installation

1. **Clone the repository**:
   ```
   git clone https://github.com/yourusername/bioinformatics-semantic-search.git
   cd bioinformatics-semantic-search
   ```

2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Copy `.env.example` to `.env` and add your API keys:
   ```
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Start Qdrant**:
   ```
   # Using Docker (recommended)
   docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
   ```

## Usage

### Basic Workflow

1. **Create a collection**:
   ```
   python qdrant_db/create_collection.py
   ```

2. **Upload tool data**:
   ```
   python qdrant_db/upload_data.py
   ```

3. **Query for tools**:
   ```
   python qdrant_db/query_data.py
   ```

### LangChain Integration

We've integrated with LangChain to enhance capabilities:

1. **Using LangChain with existing data**:
   ```
   python qdrant_db/langchain_integration.py
   ```

2. **Loading data with LangChain format**:
   ```
   python qdrant_db/langchain_data_loader.py
   ```

3. **Interactive Bioinformatics Assistant**:
   ```
   python bioinfo_assistant.py
   ```

## LangChain Features

This project leverages LangChain in several ways:

1. **Vector Store Integration**: Clean abstraction over Qdrant through LangChain's vector store classes

2. **Retrieval Augmented Generation (RAG)**: Combines retrieval with language models for more accurate and informative responses

3. **Conversational Memory**: Maintain context across multiple questions for a more natural interaction

4. **Unified Interface**: Consistent patterns for working with embeddings, documents, and language models

## Adding More Tools

To expand the database with more bioinformatics tools:

1. Edit `qdrant_db/upload_data.py` or `qdrant_db/langchain_data_loader.py`
2. Add new tool entries to the `bioinformatics_tools` list
3. Run the script to upload the new data

## Project Structure

- `qdrant_db/`: Core functionality using direct Qdrant client
  - `create_collection.py`: Creates the vector database collection
  - `upload_data.py`: Uploads tool data to Qdrant
  - `query_data.py`: Simple query interface
  - `langchain_integration.py`: LangChain integration examples
  - `langchain_data_loader.py`: Loading data in LangChain format
- `bioinfo_assistant.py`: Interactive assistant using LangChain
- `requirements.txt`: Project dependencies
- `.env.example`: Template for environment variables

## Future Improvements

- Web interface for easier interaction
- Support for more types of bioinformatics resources
- Integration with other knowledge bases and scientific literature
- Specialized agents for different bioinformatics subdomains
