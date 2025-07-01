#!/usr/bin/env python3
"""
Test script for the Unified Bioinformatics Search Engine
"""

import asyncio
from unified_bioinformatics_search import UnifiedBioinformaticsSearch

async def test_unified_system():
    """Test the unified search system with a sample query"""
    
    print("🧪 TESTING UNIFIED BIOINFORMATICS SEARCH SYSTEM")
    print("=" * 60)
    
    # Initialize the search engine
    search_engine = UnifiedBioinformaticsSearch()
    
    # Show which systems are available
    print("\n📊 System Availability:")
    for system, status in search_engine.systems_status.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {system.upper()}: {'Available' if status else 'Not configured'}")
    
    available_count = sum(search_engine.systems_status.values())
    print(f"\n📈 Total: {available_count}/4 systems available")
    
    if available_count == 0:
        print("\n❌ No systems available. Please configure at least one system.")
        return
    
    # Test with a sample query
    test_query = "protein sequence alignment tools"
    print(f"\n🔍 Testing with query: '{test_query}'")
    print("-" * 60)
    
    try:
        result = await search_engine.comprehensive_search(test_query, max_tools=3)
        
        print(f"\n✅ SEARCH COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Execution time: {result.execution_time:.2f} seconds")
        print(f"🎯 Confidence score: {result.confidence_score:.2f}")
        print(f"🔧 Systems used: {', '.join(result.execution_summary['systems_used'])}")
        print(f"🛠️  Tools found: {len(result.top_recommendations)}")
        
        if result.top_recommendations:
            print(f"\n🎯 TOP RECOMMENDATIONS:")
            for i, tool in enumerate(result.top_recommendations, 1):
                print(f"  {i}. {tool.name} (source: {tool.source}, score: {tool.relevance_score:.2f})")
        
        if result.execution_summary.get("errors"):
            print(f"\n⚠️  Errors encountered:")
            for error in result.execution_summary["errors"]:
                print(f"  • {error}")
        
        print(f"\n🎉 TEST COMPLETED - Your unified system is working!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_unified_system()) 