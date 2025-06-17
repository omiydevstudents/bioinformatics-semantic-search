from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchText
from qdrant_client.http.exceptions import UnexpectedResponse

from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import uuid
import json

def create_embedding_text(tool_data):
    base_text = tool_data["text"]
    topics = tool_data["metadata"].get("topics", [])
    operations = tool_data["metadata"].get("operations", [])
    
    enhanced_text = base_text
    
    if topics:
        enhanced_text += f"\nTopics: {', '.join(topics)}"
    
    if operations:
        enhanced_text += f"\nOperations: {', '.join(operations)}"
    
    return enhanced_text

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Get Qdrant credentials from environment variables
    api_key = os.getenv("QDRANT_API_KEY")
    cluster_url = os.getenv("QDRANT_CLUSTER_URL")

    # Connect to Qdrant cluster if credentials are available, otherwise use local
    if api_key and cluster_url:
        client = QdrantClient(url=cluster_url, api_key=api_key)
        print(f"Connected to Qdrant cloud cluster at {cluster_url}")
    else:
        client = QdrantClient(url="http://localhost:6333")
        print("Connected to local Qdrant instance")

    colName = "OmiyDB"

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(script_dir)
    json_file_path = os.path.join(script_dir, 'biotools_python_tools.json')
    print(json_file_path)

    with open(json_file_path, 'r', encoding='utf-8') as file: # is an array of json objects
        bioinformatics_tools = json.load(file)

    model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    model = SentenceTransformer(model_name)
    print(f"Model '{model_name}' loaded successfully!")

    # Convert descriptions to embeddings and upload to Qdrant
    tool_ids = [] # e.g. 0a1ab736-8765-4900-827b-7dc6ebfd8f2b
    tool_vectors = [] # e.g. [-0.04451117664575577, 0.07700273394584656, ..., 0.3271920084953308]
    tool_payloads = [] # e.g. {'tool_name': 'BioPython', 'description': 'A set of freely ... for developers.', 'url': 'https://biopython.org/'}

    # Process each tool
    for tool in bioinformatics_tools:
        tool_name = tool['metadata']['name']
        
        # Check if the tool already exists in the collection
        try:
            existing_tools = client.scroll(
                collection_name=colName,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="name",
                            match=MatchText(text=tool_name)
                        )
                    ]
                ),
                limit=1
            )[0]
            
            if existing_tools:
                print(f"Tool '{tool_name}' already exists in the collection. Skipping.")
                continue
        except UnexpectedResponse:
            # Collection might not exist yet or other API issue
            pass
        
        # Generate a unique ID
        tool_id = str(uuid.uuid4())
        tool_ids.append(tool_id)
        
        # Create vector embedding using the enhanced text
        text_to_embed = create_embedding_text(tool)
        vector = model.encode(text_to_embed).tolist()
        tool_vectors.append(vector)
        
        # Add payload with all tool data
        payload = tool['metadata'].copy()
        payload['description'] = tool['text']  # Add the descriptive text
        tool_payloads.append(payload)

    # Only upload if we have tools to upload
    if tool_ids:
        # Upload in batches to avoid timeout
        batch_size = 50
        total_uploaded = 0
        
        for i in range(0, len(tool_ids), batch_size):
            batch_ids = tool_ids[i:i+batch_size]
            batch_vectors = tool_vectors[i:i+batch_size]
            batch_payloads = tool_payloads[i:i+batch_size]
            
            try:
                client.upsert(
                    collection_name=colName,
                    points=[
                        PointStruct(
                            id=tool_id, 
                            vector=vector, 
                            payload=payload
                        )
                        for tool_id, vector, payload in zip(batch_ids, batch_vectors, batch_payloads)
                    ]
                )
                total_uploaded += len(batch_ids)
                print(f"Uploaded batch {i//batch_size + 1}: {len(batch_ids)} tools (Total: {total_uploaded})")
                
            except Exception as e:
                print(f"Error uploading batch {i//batch_size + 1}: {e}")
                continue
        
        print(f"Successfully uploaded {total_uploaded} new bioinformatics tools to the collection '{colName}'")
    else:
        print("No new tools to upload. All tools already exist in the collection.")

if __name__ == "__main__":
    main()