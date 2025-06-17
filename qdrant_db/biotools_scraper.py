#!/usr/bin/env python3
"""
Bio.tools Python Tools Extractor

This script demonstrates how to:
1. Connect to the Bio.tools API
2. Filter for Python tools specifically  
3. Iterate through all pages of results
4. Extract tool names and descriptions
5. Prepare data for storage in a vector database (like Qdrant)

Author: Assistant
"""

import requests
import time
import json
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class BioinformaticsTool:
    """Data class to store tool information"""
    name: str
    description: str
    biotools_id: str
    homepage: Optional[str] = None
    language: List[str] = None
    topics: List[str] = None
    operations: List[str] = None

    def __post_init__(self):
        if self.language is None:
            self.language = []
        if self.topics is None:
            self.topics = []
        if self.operations is None:
            self.operations = []


class BioToolsAPI:
    """Class to interact with Bio.tools API"""
    
    BASE_URL = "https://bio.tools/api/tool/"
    
    def __init__(self, requests_per_second: float = 1.0):
        """
        Initialize the API client
        
        Args:
            requests_per_second: Rate limiting for API calls
        """
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'BioTools-Python-Extractor/1.0'
        })
        self.delay = 1.0 / requests_per_second
        
    def _make_request(self, url: str, params: Dict) -> Optional[Dict]:
        """
        Make a rate-limited API request
        
        Args:
            url: API endpoint URL
            params: Query parameters
            
        Returns:
            JSON response data or None if failed
        """
        try:
            time.sleep(self.delay)  # Rate limiting
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return None
    
    def get_python_tools(self, page_size: int = 100) -> List[BioinformaticsTool]:
        """
        Fetch all Python tools from Bio.tools API
        
        Args:
            page_size: Number of results per page (max 100)
            
        Returns:
            List of BioinformaticsTool objects
        """
        tools = []
        page = 1
        
        print(f"🔍 Starting to fetch Python tools from Bio.tools API...")
        print(f"📄 Using page size: {page_size}")
        
        while True:
            print(f"📖 Fetching page {page}...")
            
            params = {
                'language': 'Python',
                'page': page,
                'format': 'json',
                'page_size': min(page_size, 100)  # API max is 100
            }
            
            data = self._make_request(self.BASE_URL, params)
            
            if not data:
                print(f"❌ Failed to fetch page {page}")
                break
                
            # Parse the response
            total_count = data.get('count', 0)
            current_tools = data.get('list', [])
            
            if page == 1:
                print(f"📊 Total Python tools found: {total_count}")
            
            if not current_tools:
                print(f"✅ No more tools found. Finished at page {page-1}")
                break
                
            # Process each tool
            for tool_data in current_tools:
                tool = self._parse_tool(tool_data)
                if tool:
                    tools.append(tool)
            
            print(f"✅ Page {page}: Found {len(current_tools)} tools")
            
            # Check if there are more pages
            if not data.get('next'):
                print(f"🏁 Reached last page ({page})")
                break
                
            page += 1
            
            # Safety break to avoid infinite loops during testing
            if page > 50:  # Remove this limit for full extraction
                print(f"⚠️  Stopping at page {page-1} for demo purposes")
                break
        
        print(f"🎉 Total tools extracted: {len(tools)}")
        return tools
    
    def _parse_tool(self, tool_data: Dict) -> Optional[BioinformaticsTool]:
        """
        Parse a single tool from API response
        
        Args:
            tool_data: Raw tool data from API
            
        Returns:
            BioinformaticsTool object or None if parsing failed
        """
        try:
            # Extract basic information
            name = tool_data.get('name', 'Unknown')
            description = tool_data.get('description', 'No description available')
            biotools_id = tool_data.get('biotoolsID', 'unknown')
            homepage = tool_data.get('homepage')
            
            # Extract language information
            languages = tool_data.get('language', [])
            
            # Extract topics
            topics = []
            for topic in tool_data.get('topic', []):
                if isinstance(topic, dict) and 'term' in topic:
                    topics.append(topic['term'])
            
            # Extract operations
            operations = []
            for func in tool_data.get('function', []):
                if isinstance(func, dict):
                    for op in func.get('operation', []):
                        if isinstance(op, dict) and 'term' in op:
                            operations.append(op['term'])
            
            return BioinformaticsTool(
                name=name,
                description=description,
                biotools_id=biotools_id,
                homepage=homepage,
                language=languages,
                topics=topics,
                operations=operations
            )
        except Exception as e:
            print(f"⚠️  Failed to parse tool: {e}")
            return None


def main():
    """Main function to demonstrate the tool extraction"""
    
    print("🧬 Bio.tools Python Tools Extractor")
    print("=" * 50)
    
    # Initialize API client
    api = BioToolsAPI(requests_per_second=2.0)  # Be nice to their servers
    
    # Fetch Python tools
    python_tools = api.get_python_tools(page_size=50)
    
    print("\n" + "=" * 50)
    print("📋 EXTRACTION RESULTS")
    print("=" * 50)
    
    if not python_tools:
        print("❌ No Python tools found or API request failed")
        return
    
    # Display summary
    print(f"✅ Successfully extracted {len(python_tools)} Python tools")
    print(f"📊 Sample of tools found:")
    print("-" * 30)
    
    # Show first 10 tools as examples
    for i, tool in enumerate(python_tools[:10], 1):
        print(f"\n{i:2d}. {tool.name}")
        print(f"    ID: {tool.biotools_id}")
        
        # Truncate long descriptions
        desc = tool.description[:100] + "..." if len(tool.description) > 100 else tool.description
        print(f"    Description: {desc}")
        
        if tool.homepage:
            print(f"    Homepage: {tool.homepage}")
        
        if tool.topics:
            print(f"    Topics: {', '.join(tool.topics[:3])}")
        
        if tool.operations:
            print(f"    Operations: {', '.join(tool.operations[:2])}")
    
    if len(python_tools) > 10:
        print(f"\n... and {len(python_tools) - 10} more tools")
    
    # Demonstrate data preparation for vector database
    print("\n" + "=" * 50)
    print("🗄️  QDRANT DATABASE PREPARATION")
    print("=" * 50)
    
    # Prepare data for Qdrant storage
    qdrant_data = []
    for tool in python_tools:
        # Combine name and description for embedding
        combined_text = f"{tool.name}. {tool.description}"
        
        # Prepare metadata
        metadata = {
            "name": tool.name,
            "biotools_id": tool.biotools_id,
            "homepage": tool.homepage,
            "language": tool.language,
            "topics": tool.topics,
            "operations": tool.operations
        }
        
        qdrant_entry = {
            "text": combined_text,
            "metadata": metadata
        }
        
        qdrant_data.append(qdrant_entry)
    
    print(f"✅ Prepared {len(qdrant_data)} entries for Qdrant storage")
    print("\n📝 Sample Qdrant entry:")
    print(json.dumps(qdrant_data[0], indent=2))
    
    # Save to JSON file for later use
    output_file = "biotools_python_tools.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(qdrant_data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Data saved to {output_file}")
        print(f"   Ready for import into Qdrant vector database!")
    except Exception as e:
        print(f"❌ Failed to save data: {e}")
    
    print("\n🎯 NEXT STEPS:")
    print("1. Use this data to populate your Qdrant vector database")
    print("2. Generate embeddings for the 'text' field using HuggingFace models")
    print("3. Store embeddings with metadata for semantic search")
    print("4. Build your LLM-powered tool recommendation system!")


if __name__ == "__main__":
    main()