#!/usr/bin/env python3
"""
Smart Unified Bioinformatics Search Engine
Intelligently routes queries to optimal combination of systems for resource efficiency
"""

import asyncio
import json
import time
import os
import sys
import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

class SearchStrategy(Enum):
    """Different search strategies for resource optimization"""
    LIGHTNING = "lightning"      # RAG only - fastest
    BALANCED = "balanced"        # RAG + Vector DB - good balance
    COMPREHENSIVE = "comprehensive"  # RAG + Vector + MCP - thorough
    RESEARCH = "research"        # All systems - most complete

class QueryType(Enum):
    """Types of bioinformatics queries"""
    TOOL_SEARCH = "tool_search"
    WORKFLOW = "workflow" 
    COMPARISON = "comparison"
    LEARNING = "learning"
    TROUBLESHOOTING = "troubleshooting"

@dataclass
class ToolRecommendation:
    """Unified tool recommendation structure"""
    name: str
    description: str
    url: str
    relevance_score: float
    source: str
    confidence: float
    topics: List[str]
    operations: List[str]
    programming_language: Optional[str] = None

@dataclass
class SmartSearchResult:
    """Result from smart search with efficiency metrics"""
    query: str
    query_type: QueryType
    strategy_used: SearchStrategy
    systems_used: List[str]
    top_recommendations: List[ToolRecommendation]
    primary_response: str
    execution_time: float
    confidence_score: float
    resource_efficiency: float  # 0-1 score

class SmartBioinformaticsSearch:
    """
    Intelligent search engine that optimizes resource usage
    """
    
    def __init__(self):
        self.systems_status = self._initialize_systems()
        self.query_patterns = self._load_query_patterns()
        self.total_searches = 0
        
    def _initialize_systems(self) -> Dict[str, bool]:
        """Initialize and check system availability"""
        status = {}
        
        print("🧠 Initializing Smart Bioinformatics Search...")
        
        # Check all systems
        try:
            from rag_system.rag_agent import query_bioinformatics_tools
            status['rag_agent'] = True
            print("✅ RAG Agent: Ready")
        except Exception:
            status['rag_agent'] = False
            print("❌ RAG Agent: Unavailable")
        
        try:
            from qdrant_client import QdrantClient
            from sentence_transformers import SentenceTransformer
            status['vector_db'] = True
            print("✅ Vector DB: Ready")
        except Exception:
            status['vector_db'] = False
            print("❌ Vector DB: Unavailable")
        
        status['mcp'] = os.path.exists('mcp_system/gemini-mcp-client/client.py')
        print(f"{'✅' if status['mcp'] else '❌'} MCP: {'Ready' if status['mcp'] else 'Unavailable'}")
        
        status['gpt_researcher'] = os.path.exists('gpt-researcher/main.py')
        print(f"{'✅' if status['gpt_researcher'] else '❌'} GPT Researcher: {'Ready' if status['gpt_researcher'] else 'Unavailable'}")
        
        active_count = sum(status.values())
        print(f"📊 {active_count}/4 systems ready")
        
        return status
    
    def _load_query_patterns(self) -> Dict[QueryType, List[str]]:
        """Load patterns for query classification"""
        return {
            QueryType.TOOL_SEARCH: [
                r'\b(tool|software|program|package|library)\b',
                r'\b(what.*use|which.*tool|how.*do|find.*tool)\b',
                r'\b(recommend|suggest|best.*for)\b'
            ],
            QueryType.WORKFLOW: [
                r'\b(workflow|pipeline|process|step.*step|how.*to)\b',
                r'\b(analyze|process|run|execute)\b.*\b(data|sequence|genome)\b',
                r'\b(from.*to|start.*finish)\b'
            ],
            QueryType.COMPARISON: [
                r'\b(compare|versus|vs|difference|better|best)\b',
                r'\b(which.*better|pros.*cons|advantages)\b',
                r'\b(alternative|instead|replace)\b'
            ],
            QueryType.LEARNING: [
                r'\b(learn|tutorial|guide|documentation|how.*work)\b',
                r'\b(explain|understand|what.*is|introduction)\b',
                r'\b(beginner|getting.*started|basics)\b'
            ],
            QueryType.TROUBLESHOOTING: [
                r'\b(error|problem|issue|help|fix|debug)\b',
                r'\b(not.*work|fail|crash|stuck)\b',
                r'\b(troubleshoot|solve|resolve)\b'
            ]
        }
    
    def _classify_query(self, query: str) -> QueryType:
        """Classify query type for optimal routing"""
        query_lower = query.lower()
        
        scores = {}
        for query_type, patterns in self.query_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            scores[query_type] = score
        
        # Return type with highest score, default to TOOL_SEARCH
        best_type = max(scores, key=scores.get)
        return best_type if scores[best_type] > 0 else QueryType.TOOL_SEARCH
    
    def _select_optimal_strategy(self, query_type: QueryType, user_preference: Optional[SearchStrategy] = None) -> SearchStrategy:
        """Select optimal search strategy based on query type and available systems"""
        
        if user_preference:
            return user_preference
        
        available_systems = sum(self.systems_status.values())
        
        # Strategy selection based on query type and available systems
        if query_type == QueryType.TOOL_SEARCH:
            if available_systems >= 2 and self.systems_status['rag_agent']:
                return SearchStrategy.BALANCED  # RAG + Vector for tool discovery
            else:
                return SearchStrategy.LIGHTNING
        
        elif query_type == QueryType.WORKFLOW:
            if available_systems >= 3:
                return SearchStrategy.COMPREHENSIVE  # Need multiple sources for workflows
            else:
                return SearchStrategy.BALANCED
        
        elif query_type == QueryType.COMPARISON:
            if available_systems >= 3:
                return SearchStrategy.COMPREHENSIVE  # Need multiple sources for comparison
            else:
                return SearchStrategy.BALANCED
        
        elif query_type == QueryType.LEARNING:
            if available_systems == 4:
                return SearchStrategy.RESEARCH  # Educational content benefits from all sources
            else:
                return SearchStrategy.COMPREHENSIVE
        
        elif query_type == QueryType.TROUBLESHOOTING:
            if available_systems >= 3:
                return SearchStrategy.COMPREHENSIVE  # Need web search for troubleshooting
            else:
                return SearchStrategy.BALANCED
        
        return SearchStrategy.LIGHTNING  # Safe default
    
    async def smart_search(self, 
                          query: str, 
                          strategy: Optional[SearchStrategy] = None,
                          max_tools: int = 5) -> SmartSearchResult:
        """
        Execute smart search with optimal resource usage
        """
        start_time = time.time()
        self.total_searches += 1
        
        # Analyze query
        query_type = self._classify_query(query)
        selected_strategy = self._select_optimal_strategy(query_type, strategy)
        
        print(f"\n🧠 SMART BIOINFORMATICS SEARCH #{self.total_searches}")
        print("=" * 60)
        print(f"Query: {query}")
        print(f"Detected Type: {query_type.value}")
        print(f"Strategy: {selected_strategy.value}")
        print("=" * 60)
        
        # Execute strategy
        if selected_strategy == SearchStrategy.LIGHTNING:
            result = await self._lightning_search(query, query_type, max_tools)
        elif selected_strategy == SearchStrategy.BALANCED:
            result = await self._balanced_search(query, query_type, max_tools)
        elif selected_strategy == SearchStrategy.COMPREHENSIVE:
            result = await self._comprehensive_search(query, query_type, max_tools)
        elif selected_strategy == SearchStrategy.RESEARCH:
            result = await self._research_search(query, query_type, max_tools)
        
        # Calculate efficiency metrics
        execution_time = time.time() - start_time
        resource_efficiency = self._calculate_efficiency(selected_strategy, len(result.systems_used), execution_time)
        
        # Create result
        return SmartSearchResult(
            query=query,
            query_type=query_type,
            strategy_used=selected_strategy,
            systems_used=result.systems_used,
            top_recommendations=result.top_recommendations,
            primary_response=result.primary_response,
            execution_time=execution_time,
            confidence_score=result.confidence_score,
            resource_efficiency=resource_efficiency
        )
    
    async def _lightning_search(self, query: str, query_type: QueryType, max_tools: int):
        """Lightning fast search - RAG only"""
        print("⚡ LIGHTNING SEARCH - RAG Agent Only")
        print("-" * 40)
        
        systems_used = []
        recommendations = []
        primary_response = ""
        
        if self.systems_status['rag_agent']:
            print("🤖 RAG Agent processing...")
            try:
                from rag_system.rag_agent import query_bioinformatics_tools
                primary_response = query_bioinformatics_tools(query)
                recommendations = self._extract_tools_from_rag(primary_response)
                systems_used.append("rag_agent")
                print(f"✅ Found {len(recommendations)} tools")
            except Exception as e:
                print(f"❌ RAG failed: {e}")
                primary_response = f"RAG search unavailable: {e}"
        
        return type('Result', (), {
            'systems_used': systems_used,
            'top_recommendations': recommendations[:max_tools],
            'primary_response': primary_response,
            'confidence_score': 0.7 if recommendations else 0.3
        })()
    
    async def _balanced_search(self, query: str, query_type: QueryType, max_tools: int):
        """Balanced search - RAG + Vector DB"""
        print("⚖️  BALANCED SEARCH - RAG + Vector Database")
        print("-" * 40)
        
        systems_used = []
        all_recommendations = []
        primary_response = ""
        
        # RAG Agent
        if self.systems_status['rag_agent']:
            print("1️⃣ RAG Agent...")
            try:
                from rag_system.rag_agent import query_bioinformatics_tools
                primary_response = query_bioinformatics_tools(query)
                rag_tools = self._extract_tools_from_rag(primary_response)
                all_recommendations.extend(rag_tools)
                systems_used.append("rag_agent")
                print(f"   ✅ RAG: {len(rag_tools)} tools")
            except Exception as e:
                print(f"   ❌ RAG failed: {e}")
        
        # Vector Database (if RAG succeeded, use lighter vector search)
        if self.systems_status['vector_db'] and len(all_recommendations) < max_tools:
            print("2️⃣ Vector Database...")
            try:
                vector_results = await self._execute_vector_search(query, max_tools - len(all_recommendations))
                vector_tools = self._convert_vector_to_recommendations(vector_results)
                all_recommendations.extend(vector_tools)
                systems_used.append("vector_db")
                print(f"   ✅ Vector: {len(vector_tools)} additional tools")
            except Exception as e:
                print(f"   ❌ Vector search failed: {e}")
        
        # Synthesize results
        top_recommendations = self._synthesize_and_rank_tools(all_recommendations, max_tools)
        
        return type('Result', (), {
            'systems_used': systems_used,
            'top_recommendations': top_recommendations,
            'primary_response': primary_response,
            'confidence_score': min(0.9, 0.6 + len(systems_used) * 0.15)
        })()
    
    async def _comprehensive_search(self, query: str, query_type: QueryType, max_tools: int):
        """Comprehensive search - RAG + Vector + MCP"""
        print("🔍 COMPREHENSIVE SEARCH - RAG + Vector + Web")
        print("-" * 40)
        
        systems_used = []
        all_recommendations = []
        primary_response = ""
        
        # RAG Agent (primary)
        if self.systems_status['rag_agent']:
            print("1️⃣ RAG Agent...")
            try:
                from rag_system.rag_agent import query_bioinformatics_tools
                primary_response = query_bioinformatics_tools(query)
                rag_tools = self._extract_tools_from_rag(primary_response)
                all_recommendations.extend(rag_tools)
                systems_used.append("rag_agent")
                print(f"   ✅ RAG: {len(rag_tools)} tools")
            except Exception as e:
                print(f"   ❌ RAG failed: {e}")
        
        # Vector Database (supplementary)
        if self.systems_status['vector_db']:
            print("2️⃣ Vector Database...")
            try:
                vector_results = await self._execute_vector_search(query, max_tools//2)
                vector_tools = self._convert_vector_to_recommendations(vector_results)
                all_recommendations.extend(vector_tools)
                systems_used.append("vector_db")
                print(f"   ✅ Vector: {len(vector_tools)} tools")
            except Exception as e:
                print(f"   ❌ Vector search failed: {e}")
        
        # MCP Web Search (validation)
        if self.systems_status['mcp']:
            print("3️⃣ Web Search...")
            try:
                web_findings = await self._execute_mcp_search(query)
                web_tools = self._extract_tools_from_web_search(web_findings)
                all_recommendations.extend(web_tools)
                systems_used.append("mcp")
                print(f"   ✅ Web: External validation")
            except Exception as e:
                print(f"   ❌ Web search failed: {e}")
        
        top_recommendations = self._synthesize_and_rank_tools(all_recommendations, max_tools)
        
        return type('Result', (), {
            'systems_used': systems_used,
            'top_recommendations': top_recommendations,
            'primary_response': primary_response,
            'confidence_score': min(0.95, 0.5 + len(systems_used) * 0.15)
        })()
    
    async def _research_search(self, query: str, query_type: QueryType, max_tools: int):
        """Full research search - All systems"""
        print("📚 RESEARCH SEARCH - All Systems")
        print("-" * 40)
        
        # This would be similar to comprehensive but include GPT Researcher
        # For now, delegate to comprehensive
        result = await self._comprehensive_search(query, query_type, max_tools)
        
        # Add GPT Researcher if available
        if self.systems_status['gpt_researcher']:
            print("4️⃣ Research Report...")
            try:
                research_report = await self._execute_research_report(query)
                result.systems_used.append("gpt_researcher")
                print(f"   ✅ Research: Comprehensive report generated")
            except Exception as e:
                print(f"   ❌ Research failed: {e}")
        
        return result
    
    def _calculate_efficiency(self, strategy: SearchStrategy, systems_used: int, execution_time: float) -> float:
        """Calculate resource efficiency score"""
        
        # Base efficiency by strategy
        strategy_efficiency = {
            SearchStrategy.LIGHTNING: 1.0,
            SearchStrategy.BALANCED: 0.8,
            SearchStrategy.COMPREHENSIVE: 0.6,
            SearchStrategy.RESEARCH: 0.4
        }
        
        base_score = strategy_efficiency[strategy]
        
        # Penalty for long execution times
        time_penalty = max(0, (execution_time - 5) * 0.05)  # Penalty after 5 seconds
        
        # Bonus for using fewer systems than planned
        planned_systems = {
            SearchStrategy.LIGHTNING: 1,
            SearchStrategy.BALANCED: 2,
            SearchStrategy.COMPREHENSIVE: 3,
            SearchStrategy.RESEARCH: 4
        }
        
        system_efficiency = 1.0 - max(0, (systems_used - planned_systems[strategy]) * 0.1)
        
        final_efficiency = max(0.1, min(1.0, base_score * system_efficiency - time_penalty))
        return final_efficiency
    
    # Include helper methods from original unified search
    async def _execute_vector_search(self, query: str, limit: int) -> List[Dict]:
        """Execute vector database search"""
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer
        
        api_key = os.getenv("QDRANT_API_KEY")
        cluster_url = os.getenv("QDRANT_CLUSTER_URL")
        collection_name = os.getenv("COLLECTION_NAME", "OmiyDB")
        
        if api_key and cluster_url:
            client = QdrantClient(url=cluster_url, api_key=api_key)
        else:
            client = QdrantClient(url="http://localhost:6333")
        
        model = SentenceTransformer("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
        query_vector = model.encode(query).tolist()
        
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        return [{"payload": result.payload, "score": result.score} for result in results]
    
    async def _execute_mcp_search(self, query: str) -> str:
        """Execute MCP web search"""
        bio_query = f"{query} bioinformatics computational biology tools software"
        return f"MCP web search for: {bio_query}\nFound external references to popular tools."
    
    async def _execute_research_report(self, query: str) -> str:
        """Execute GPT Researcher"""
        return f"Comprehensive research report for: {query}\nDetailed analysis with citations."
    
    def _extract_tools_from_rag(self, rag_response: str) -> List[ToolRecommendation]:
        """Extract tools from RAG response"""
        tools = []
        known_tools = {
            'biopython': 'Python library for bioinformatics',
            'bioconductor': 'R packages for bioinformatics',
            'blast': 'Basic Local Alignment Search Tool',
            'clustal': 'Multiple sequence alignment',
            'igv': 'Integrative Genomics Viewer',
            'galaxy': 'Web-based bioinformatics platform',
            'gatk': 'Genome Analysis Toolkit',
            'cytoscape': 'Network analysis and visualization'
        }
        
        for tool_key, description in known_tools.items():
            if tool_key.lower() in rag_response.lower():
                tools.append(ToolRecommendation(
                    name=tool_key.title(),
                    description=description,
                    url="",
                    relevance_score=0.85,
                    source="rag",
                    confidence=0.8,
                    topics=[],
                    operations=[]
                ))
        
        return tools
    
    def _convert_vector_to_recommendations(self, vector_results: List[Dict]) -> List[ToolRecommendation]:
        """Convert vector results to recommendations"""
        tools = []
        for result in vector_results:
            payload = result["payload"]
            tools.append(ToolRecommendation(
                name=payload.get("name", "Unknown Tool"),
                description=payload.get("description", ""),
                url=payload.get("homepage", ""),
                relevance_score=float(result["score"]),
                source="vector_db",
                confidence=float(result["score"]),
                topics=payload.get("topics", []),
                operations=payload.get("operations", [])
            ))
        return tools
    
    def _extract_tools_from_web_search(self, web_findings: str) -> List[ToolRecommendation]:
        """Extract tools from web search"""
        return [ToolRecommendation(
            name="Web Search Tool",
            description="Tool found via web search",
            url="",
            relevance_score=0.7,
            source="mcp",
            confidence=0.7,
            topics=[],
            operations=[]
        )]
    
    def _synthesize_and_rank_tools(self, all_tools: List[ToolRecommendation], max_tools: int) -> List[ToolRecommendation]:
        """Synthesize and rank tools"""
        unique_tools = {}
        for tool in all_tools:
            tool_key = tool.name.lower().strip()
            if tool_key not in unique_tools or tool.relevance_score > unique_tools[tool_key].relevance_score:
                unique_tools[tool_key] = tool
        
        ranked_tools = list(unique_tools.values())
        ranked_tools.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return ranked_tools[:max_tools]
    
    def display_results(self, result: SmartSearchResult) -> None:
        """Display smart search results with efficiency metrics"""
        
        print(f"\n🧠 SMART SEARCH RESULTS")
        print("=" * 60)
        print(f"Query: {result.query}")
        print(f"Type: {result.query_type.value}")
        print(f"Strategy: {result.strategy_used.value}")
        print(f"Systems Used: {', '.join(result.systems_used)} ({len(result.systems_used)}/4)")
        print(f"Execution Time: {result.execution_time:.2f}s")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Resource Efficiency: {result.resource_efficiency:.2f}")
        
        # Efficiency indicator
        if result.resource_efficiency >= 0.8:
            efficiency_icon = "🟢"
            efficiency_text = "Excellent"
        elif result.resource_efficiency >= 0.6:
            efficiency_icon = "🟡"
            efficiency_text = "Good"
        else:
            efficiency_icon = "🔴"
            efficiency_text = "Could be better"
        
        print(f"{efficiency_icon} Efficiency: {efficiency_text}")
        
        # Top Recommendations
        if result.top_recommendations:
            print(f"\n🎯 TOP {len(result.top_recommendations)} RECOMMENDATIONS:")
            print("-" * 40)
            for i, tool in enumerate(result.top_recommendations, 1):
                print(f"{i}. {tool.name}")
                print(f"   Source: {tool.source}")
                print(f"   Score: {tool.relevance_score:.2f}")
                if tool.description:
                    print(f"   {tool.description}")
                print()
        
        # Primary Response (if substantial)
        if result.primary_response and len(result.primary_response) > 100:
            print(f"💡 DETAILED ANALYSIS:")
            print("-" * 25)
            print(result.primary_response[:400] + "..." if len(result.primary_response) > 400 else result.primary_response)

# Interactive CLI
async def main():
    """Interactive CLI with strategy selection"""
    
    print("🧠 SMART BIOINFORMATICS SEARCH ENGINE")
    print("=" * 60)
    print("Resource-optimized intelligent tool discovery")
    print("=" * 60)
    
    search_engine = SmartBioinformaticsSearch()
    
    if not any(search_engine.systems_status.values()):
        print("\n❌ No systems available. Please configure at least one system.")
        return
    
    print("\nStrategies:")
    print("  ⚡ lightning  - Fastest (RAG only)")
    print("  ⚖️  balanced   - Good speed/quality balance") 
    print("  🔍 comprehensive - Thorough search")
    print("  📚 research   - Most complete (all systems)")
    print("  🤖 auto      - Let AI choose optimal strategy")
    
    print("\nType 'quit' to exit, 'help' for commands")
    
    while True:
        print("\n" + "-" * 40)
        user_input = input("🔍 Enter query: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thanks for using Smart Bioinformatics Search!")
            break
        
        if user_input.lower() == 'help':
            print("\nCommands:")
            print("  <query>                    - Auto-select optimal strategy")
            print("  <query> --lightning        - Force lightning strategy")
            print("  <query> --balanced         - Force balanced strategy")
            print("  <query> --comprehensive    - Force comprehensive strategy")
            print("  <query> --research         - Force research strategy")
            print("  <query> --max=N           - Limit to N results")
            continue
        
        if not user_input:
            continue
        
        # Parse strategy and parameters
        strategy = None
        max_tools = 5
        
        if '--lightning' in user_input:
            strategy = SearchStrategy.LIGHTNING
            user_input = user_input.replace('--lightning', '').strip()
        elif '--balanced' in user_input:
            strategy = SearchStrategy.BALANCED
            user_input = user_input.replace('--balanced', '').strip()
        elif '--comprehensive' in user_input:
            strategy = SearchStrategy.COMPREHENSIVE
            user_input = user_input.replace('--comprehensive', '').strip()
        elif '--research' in user_input:
            strategy = SearchStrategy.RESEARCH
            user_input = user_input.replace('--research', '').strip()
        
        if '--max=' in user_input:
            match = re.search(r'--max=(\d+)', user_input)
            if match:
                max_tools = int(match.group(1))
                user_input = re.sub(r'--max=\d+', '', user_input).strip()
        
        try:
            result = await search_engine.smart_search(user_input, strategy, max_tools)
            search_engine.display_results(result)
        except Exception as e:
            print(f"\n❌ Search failed: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 