#!/usr/bin/env python
"""
Test script to check if the MCP system setup is functional.
This script tests the basic components without requiring API keys.
"""

import asyncio
import os
import sys
import json
from contextlib import AsyncExitStack

# Test imports
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from langchain_mcp_adapters.tools import load_mcp_tools
    print("✓ All MCP imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_config_loading():
    """Test if the configuration file can be loaded."""
    print("\n=== Testing Configuration Loading ===")
    
    # Check if config file exists
    config_path = "theailanguage_config.json"
    if os.path.exists(config_path):
        print(f"✓ Configuration file found: {config_path}")
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            print(f"✓ Configuration loaded successfully")
            print(f"  - MCP servers: {list(config.get('mcpServers', {}).keys())}")
            return config
        except Exception as e:
            print(f"❌ Failed to load configuration: {e}")
            return None
    else:
        print(f"❌ Configuration file not found: {config_path}")
        return None

async def test_mcp_connection(config):
    """Test MCP server connections."""
    print("\n=== Testing MCP Server Connections ===")
    
    mcp_servers = config.get("mcpServers", {})
    if not mcp_servers:
        print("❌ No MCP servers found in configuration")
        return False
    
    tools = []
    
    async with AsyncExitStack() as stack:
        for server_name, server_info in mcp_servers.items():
            print(f"\n🔗 Testing connection to: {server_name}")
            print(f"  Command: {server_info.get('command', 'N/A')}")
            print(f"  Args: {server_info.get('args', [])}")
            
            try:
                # Create server parameters
                server_params = StdioServerParameters(
                    command=server_info["command"],
                    args=server_info["args"]
                )
                
                # Try to establish connection
                read, write = await stack.enter_async_context(stdio_client(server_params))
                session = await stack.enter_async_context(ClientSession(read, write))
                
                # Initialize session
                await session.initialize()
                print(f"✓ Successfully connected to {server_name}")
                
                # Try to load tools
                try:
                    server_tools = await load_mcp_tools(session)
                    print(f"✓ Loaded {len(server_tools)} tools from {server_name}")
                    tools.extend(server_tools)
                except Exception as e:
                    print(f"⚠️  Could not load tools from {server_name}: {e}")
                
            except Exception as e:
                print(f"❌ Failed to connect to {server_name}: {e}")
    
    print(f"\n📊 Summary: {len(tools)} total tools loaded")
    return len(tools) > 0

def test_environment():
    """Test environment setup."""
    print("\n=== Testing Environment Setup ===")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check if .env file exists
    env_file = ".env"
    if os.path.exists(env_file):
        print(f"✓ Environment file found: {env_file}")
    else:
        print(f"⚠️  Environment file not found: {env_file}")
        print("  You'll need to create this file with GOOGLE_API_KEY=your-key")
    
    # Check for Google API key
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        print("✓ GOOGLE_API_KEY environment variable is set")
    else:
        print("⚠️  GOOGLE_API_KEY environment variable is not set")
        print("  This is required for the full client to work")

async def main():
    """Main test function."""
    print("🧪 MCP System Setup Test")
    print("=" * 50)
    
    # Test environment
    test_environment()
    
    # Test configuration loading
    config = test_config_loading()
    if not config:
        print("\n❌ Cannot proceed without valid configuration")
        return
    
    # Test MCP connections
    success = await test_mcp_connection(config)
    
    print("\n" + "=" * 50)
    if success:
        print("✅ MCP system setup appears to be functional!")
        print("\nNext steps:")
        print("1. Set GOOGLE_API_KEY in your .env file")
        print("2. Configure real MCP servers in theailanguage_config.json")
        print("3. Run the full client: python client.py")
    else:
        print("❌ MCP system setup has issues")
        print("\nIssues to resolve:")
        print("1. Check MCP server configurations")
        print("2. Ensure MCP servers are properly installed")
        print("3. Verify server commands and arguments")

if __name__ == "__main__":
    asyncio.run(main()) 