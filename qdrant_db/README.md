# Bioinformatics Semantic Search Engine

A vector database-powered semantic search engine for AI tools in bioinformatics. This project enables intelligent discovery of bioinformatics tools based on natural language queries.

## 🧠 Project Overview

This project creates a semantic search engine for bioinformatics tools using:
- Vector embeddings from BiomedNLP-BiomedBERT
- Qdrant vector database for similarity search
- Python for data processing and querying

## 📋 Prerequisites

- Python 3.8+ 
- Docker
- Conda

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/bioinformatics-semantic-search.git
cd bioinformatics-semantic-search
```

### 2. Set up Conda environment

```bash
# Create conda environment
conda create -n biosearch python=3.10
conda activate biosearch
```

### 3. Install dependencies

```bash
pip install qdrant-client sentence-transformers numpy uuid
```

## 🐳 Docker Setup for Qdrant

### 1. Pull the Qdrant Docker image

```bash
docker pull qdrant/qdrant
```

### 2. Start Qdrant container (linux)

```bash
docker run -d -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    --name qdrant-bioinformatics \
    qdrant/qdrant
```
### 2. Start Qdrant container (Windows)
```bash
docker run -d -p 6333:6333 -p 6334:6334 -v ${PWD}/qdrant_storage:/qdrant/storage --name qdrant-bioinformatics qdrant/qdrant
```

This command:
- Creates a container named `qdrant-bioinformatics`
- Maps HTTP port to 6333 and gRPC port to 6334
- Persists data to a `qdrant_storage` folder in your current directory

### 3. Verify Qdrant is running

```bash
# Check if the container is running
docker ps

# Access Qdrant dashboard (optional)
open http://localhost:6333/dashboard
```

## 🧪 Running the Project

The project is divided into three main scripts:

### 1. Create the collection

```bash
python create_collection.py
```

This script initializes the vector database collection with cosine similarity metric.

### 2. Upload sample data

```bash
python upload_data.py
```

This script adds bioinformatics tools to the database with embeddings.

### 3. Query the database

```bash
python query_data.py
```

This script performs semantic searches to find relevant tools.

## 📁 Project Structure

```
bioinformatics-semantic-search/
├── create_collection.py      # Initialize database and model
├── upload_data.py            # Add tools to the vector database
├── query_data.py             # Search for similar tools
├── qdrant_storage/           # Database files (not committed to git)
└── README.md                 # This file
```

## 🧹 Cleanup

To stop and remove the Qdrant container:

```bash
# Stop the container
docker stop qdrant-bioinformatics

# Remove the container
docker rm qdrant-bioinformatics
```

To empty the database but keep the container:

```bash
# Stop the container
docker stop qdrant-bioinformatics

# Remove storage files
rm -rf $(pwd)/qdrant_storage/*

# Restart the container
docker start qdrant-bioinformatics
```
