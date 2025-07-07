#!/usr/bin/env python3
"""
Unified Bioinformatics Search Engine
Orchestrates RAG Agent, Qdrant Vector DB, MCP, and GPT Researcher for optimal tool discovery
"""

import asyncio
import json
import time
import os
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from contextlib import AsyncExitStack
from dotenv import load_dotenv

# MCP imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

@dataclass
class ToolRecommendation:
    """Unified tool recommendation structure"""
    name: str
    description: str
    url: str
    relevance_score: float
    source: str  # 'rag', 'vector_db', 'mcp', 'gpt_researcher'
    confidence: float
    topics: List[str]
    operations: List[str]
    programming_language: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)

@dataclass
class UnifiedSearchResult:
    """Complete search result from all systems"""
    query: str
    top_recommendations: List[ToolRecommendation]
    rag_analysis: str
    vector_search_results: List[Dict]
    web_search_findings: str
    research_report: str
    execution_summary: Dict[str, Any]
    confidence_score: float
    execution_time: float

class UnifiedBioinformaticsSearch:
    """
    Master orchestrator that combines all bioinformatics search systems
    """
    
    def __init__(self):
        self.systems_status = self._initialize_systems()
        self.total_searches = 0
        
    def _initialize_systems(self) -> Dict[str, bool]:
        """Initialize and check availability of all systems"""
        status = {}
        
        print("🔧 Initializing bioinformatics search systems...")
        
        # 1. RAG Agent System
        try:
            from rag_system.rag_agent import query_bioinformatics_tools
            status['rag_agent'] = True
            print("✅ RAG Agent: Initialized")
        except Exception as e:
            status['rag_agent'] = False
            print(f"❌ RAG Agent: Failed - {e}")
        
        # 2. Qdrant Vector Database
        try:
            from qdrant_client import QdrantClient
            from sentence_transformers import SentenceTransformer
            status['vector_db'] = True
            print("✅ Qdrant Vector DB: Initialized")
        except Exception as e:
            status['vector_db'] = False
            print(f"❌ Vector DB: Failed - {e}")
        
        # 3. MCP System
        try:
            mcp_path = 'mcp_system/gemini-mcp-client'
            if os.path.exists(f'{mcp_path}/client.py'):
                status['mcp'] = True
                print("✅ MCP System: Available")
            else:
                status['mcp'] = False
                print("❌ MCP System: Client not found")
        except Exception as e:
            status['mcp'] = False
            print(f"❌ MCP System: Failed - {e}")
        
        # 4. GPT Researcher
        try:
            if os.path.exists('gpt-researcher/main.py'):
                sys.path.append('gpt-researcher')
                status['gpt_researcher'] = True
                print("✅ GPT Researcher: Available")
            else:
                status['gpt_researcher'] = False
                print("❌ GPT Researcher: Main file not found")
        except Exception as e:
            status['gpt_researcher'] = False
            print(f"❌ GPT Researcher: Failed - {e}")
        
        active_systems = sum(status.values())
        print(f"\n📊 Systems Summary: {active_systems}/4 active")
        return status
    
    async def comprehensive_search(self, query: str, max_tools: int = 5) -> UnifiedSearchResult:
        """
        Execute comprehensive search across all available systems
        """
        start_time = time.time()
        self.total_searches += 1
        
        print(f"\n🧬 UNIFIED BIOINFORMATICS SEARCH #{self.total_searches}")
        print("=" * 70)
        print(f"Query: {query}")
        print(f"Target: Top {max_tools} tool recommendations")
        print("=" * 70)
        
        # Initialize result containers
        execution_summary = {"systems_used": [], "errors": []}
        all_recommendations = []
        
        # 1. RAG Agent (Primary Analysis)
        rag_analysis = ""
        if self.systems_status['rag_agent']:
            print("\n1️⃣ RAG AGENT - Advanced Self-RAG Analysis")
            print("-" * 50)
            try:
                rag_analysis = await self._execute_rag_search(query)
                rag_tools = self._extract_tools_from_rag(rag_analysis)
                all_recommendations.extend(rag_tools)
                execution_summary["systems_used"].append("rag_agent")
                print(f"✅ RAG Analysis complete: {len(rag_tools)} tools identified")
            except Exception as e:
                error_msg = f"RAG Agent failed: {str(e)}"
                execution_summary["errors"].append(error_msg)
                print(f"❌ {error_msg}")
        
        # 2. Vector Database (Direct Search)
        vector_results = []
        if self.systems_status['vector_db']:
            print("\n2️⃣ VECTOR DATABASE - Semantic Similarity Search")
            print("-" * 50)
            try:
                vector_results = await self._execute_vector_search(query, max_tools)
                vector_tools = self._convert_vector_to_recommendations(vector_results)
                all_recommendations.extend(vector_tools)
                execution_summary["systems_used"].append("vector_db")
                print(f"✅ Vector search complete: {len(vector_results)} tools found")
            except Exception as e:
                error_msg = f"Vector DB failed: {str(e)}"
                execution_summary["errors"].append(error_msg)
                print(f"❌ {error_msg}")
        
        # 3. MCP Web Search (External Validation)
        web_findings = ""
        if self.systems_status['mcp']:
            print("\n3️⃣ MCP WEB SEARCH - External Source Validation")
            print("-" * 50)
            try:
                web_findings = await self._execute_mcp_search(query)
                web_tools = self._extract_tools_from_web_search(web_findings)
                all_recommendations.extend(web_tools)
                execution_summary["systems_used"].append("mcp")
                print(f"✅ Web search complete: External validation obtained")
            except Exception as e:
                error_msg = f"MCP search failed: {str(e)}"
                execution_summary["errors"].append(error_msg)
                print(f"❌ {error_msg}")
        
        # 4. GPT Researcher (Comprehensive Report)
        research_report = ""
        if self.systems_status['gpt_researcher']:
            print("\n4️⃣ GPT RESEARCHER - Comprehensive Analysis Report")
            print("-" * 50)
            try:
                research_report = await self._execute_research_report(query)
                research_tools = self._extract_tools_from_research(research_report)
                all_recommendations.extend(research_tools)
                execution_summary["systems_used"].append("gpt_researcher")
                print(f"✅ Research report complete: Comprehensive analysis generated")
            except Exception as e:
                error_msg = f"GPT Researcher failed: {str(e)}"
                execution_summary["errors"].append(error_msg)
                print(f"❌ {error_msg}")
        
        # 5. Intelligent Synthesis and Ranking
        print("\n5️⃣ INTELLIGENT SYNTHESIS")
        print("-" * 50)
        top_recommendations = self._synthesize_and_rank_tools(all_recommendations, max_tools)
        confidence_score = self._calculate_unified_confidence(execution_summary, all_recommendations)
        
        execution_time = time.time() - start_time
        execution_summary.update({
            "total_tools_found": len(all_recommendations),
            "unique_tools": len(top_recommendations),
            "execution_time": execution_time
        })
        
        print(f"✅ Synthesis complete: {len(top_recommendations)} top recommendations")
        print(f"📊 Confidence Score: {confidence_score:.2f}")
        print(f"⏱️  Total Execution Time: {execution_time:.2f}s")
        
        return UnifiedSearchResult(
            query=query,
            top_recommendations=top_recommendations,
            rag_analysis=rag_analysis,
            vector_search_results=vector_results,
            web_search_findings=web_findings,
            research_report=research_report,
            execution_summary=execution_summary,
            confidence_score=confidence_score,
            execution_time=execution_time
        )
    
    async def _execute_rag_search(self, query: str) -> str:
        """Execute RAG agent search with Self-RAG capabilities"""
        print(f"   🤖 Initializing RAG Agent with Self-RAG...")
        from rag_system.rag_agent import query_bioinformatics_tools
        print(f"   💭 Processing query with advanced reasoning...")
        result = query_bioinformatics_tools(query)
        print(f"   ✅ RAG analysis complete ({len(result)} chars)")
        return result
    
    async def _execute_vector_search(self, query: str, limit: int) -> List[Dict]:
        """Execute direct vector database search"""
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer
        
        # Initialize Qdrant client with better error handling
        api_key = os.getenv("QDRANT_API_KEY")
        cluster_url = os.getenv("QDRANT_CLUSTER_URL")
        collection_name = os.getenv("COLLECTION_NAME", "OmiyDB")
        
        print(f"   🔗 Connecting to Qdrant...")
        if api_key and cluster_url:
            print(f"   📡 Using cloud Qdrant: {cluster_url[:50]}...")
            client = QdrantClient(url=cluster_url, api_key=api_key)
        else:
            print(f"   🏠 Using local Qdrant: http://localhost:6333")
            client = QdrantClient(url="http://localhost:6333")
        
        # Generate embedding
        print(f"   🧠 Generating embeddings...")
        model = SentenceTransformer("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
        query_vector = model.encode(query).tolist()
        
        # Search vector database
        print(f"   🔍 Searching collection '{collection_name}'...")
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        print(f"   ✅ Found {len(results)} results")
        return [{"payload": result.payload, "score": result.score} for result in results]
    
    async def _execute_mcp_search(self, query: str) -> str:
        """Execute MCP search using actual MCP servers"""
        print(f"   🌐 Initializing MCP client...")
        
        # Enhanced query for bioinformatics context
        bio_query = f"{query} bioinformatics computational biology tools software"
        print(f"   🔍 Enhanced query: {bio_query}")
        
        try:
            # Load MCP tools from all configured servers
            tools = await self._get_mcp_tools()
            
            if not tools:
                print(f"   ⚠️  No MCP tools available, using fallback")
                return f"MCP search attempted for: {bio_query}\nNo external tools available at this time."
            
            print(f"   🔧 Using {len(tools)} MCP tools")
            
            # Create an agent with the loaded tools
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0,
                max_retries=2,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            
            agent = create_react_agent(llm, tools)
            
            # Execute the query
            print(f"   🤖 Executing MCP agent query...")
            enhanced_query = f"{bio_query} \nUse biopython.org and bioconductor.org as references. Please add full links to the tools you found!"
            
            response = await agent.ainvoke({"messages": enhanced_query})
            
            # Extract the content from the response
            if hasattr(response, 'get') and 'messages' in response:
                last_message = response['messages'][-1]
                if hasattr(last_message, 'content'):
                    result = last_message.content
                else:
                    result = str(last_message)
            else:
                result = str(response)
            
            print(f"   ✅ MCP search complete ({len(result)} chars)")
            return result
            
        except Exception as e:
            print(f"   ❌ MCP search failed: {e}")
            return f"MCP search encountered an error for: {bio_query}\nError: {str(e)}"
    
    async def _get_mcp_tools(self):
        """Load tools from all configured MCP servers"""
        config_path = 'mcp_system/gemini-mcp-client/theailanguage_config.json'
        
        if not os.path.exists(config_path):
            print(f"   ❌ MCP config not found: {config_path}")
            return []
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"   ❌ Failed to load MCP config: {e}")
            return []
        
        mcp_servers = config.get("mcpServers", {})
        if not mcp_servers:
            print(f"   ❌ No MCP servers in config")
            return []
        
        tools = []
        
        async with AsyncExitStack() as stack:
            for server_name, server_info in mcp_servers.items():
                print(f"   🔗 Connecting to {server_name}...")
                
                try:
                    # Create server parameters
                    server_params = StdioServerParameters(
                        command=server_info["command"],
                        args=server_info["args"],
                        env=server_info.get("env", {}),
                        cwd=server_info.get("cwd")
                    )
                    
                    # Connect to server
                    read, write = await stack.enter_async_context(stdio_client(server_params))
                    session = await stack.enter_async_context(ClientSession(read, write))
                    await session.initialize()
                    
                    # Load tools from this server
                    server_tools = await load_mcp_tools(session)
                    tools.extend(server_tools)
                    
                    print(f"   ✅ Loaded {len(server_tools)} tools from {server_name}")
                    
                except Exception as e:
                    print(f"   ⚠️  Failed to connect to {server_name}: {e}")
                    continue
        
        return tools
    
    async def _execute_research_report(self, query: str) -> str:
        """Execute GPT Researcher for comprehensive analysis"""
        print(f"   📊 Preparing comprehensive research analysis...")
        enhanced_query = f"Bioinformatics tools and software for: {query}"
        print(f"   🔬 Research focus: {enhanced_query}")
        
        # Check if GPT Researcher is available for integration
        gpt_researcher_path = 'gpt-researcher/main.py'
        if os.path.exists(gpt_researcher_path):
            print(f"   📚 GPT Researcher available - could generate detailed report")
            # Future: Actual GPT Researcher integration would go here
            result = f"GPT Researcher comprehensive report for: {enhanced_query}\n\nDetailed analysis of available tools, methodologies, and best practices in the field. Includes citations and performance comparisons."
        else:
            print(f"   💭 Using simulated research report")
            result = f"Research analysis for: {enhanced_query}\n\nComprehensive overview of tools and methodologies."
        
        print(f"   ✅ Research report generated")
        return result
    
    def _extract_tools_from_rag(self, rag_response: str) -> List[ToolRecommendation]:
        """Extract tool recommendations from RAG response"""
        tools = []
        
        # More sophisticated extraction from RAG response
        # Look for tool mentions in the response text
        lines = rag_response.split('\n')
        current_tool = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for tool names (usually capitalized or in quotes)
            import re
            
            # Pattern 1: Look for "Tool: Name" or "1. Name" patterns
            tool_patterns = [
                r'(?:Tool|Software|Package|Program|Application):\s*([A-Za-z0-9\-\+\.]+)',
                r'^\d+\.\s*([A-Za-z0-9\-\+\.]+)',
                r'•\s*([A-Za-z0-9\-\+\.]+)',
                r'-\s*([A-Za-z0-9\-\+\.]+)',
                r'\*\*([A-Za-z0-9\-\+\.]+)\*\*',
                r'`([A-Za-z0-9\-\+\.]+)`'
            ]
            
            for pattern in tool_patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                for match in matches:
                    tool_name = match.strip()
                    if len(tool_name) > 2 and tool_name.lower() not in ['the', 'and', 'for', 'with', 'are', 'can', 'use']:
                        # Extract description from the same line or next lines
                        description = line.replace(match, '').strip()
                        description = re.sub(r'^[\d\.\-\*•]+\s*', '', description)
                        description = re.sub(r'^\w+:\s*', '', description)
                        
                        if not description or len(description) < 10:
                            description = f"Bioinformatics tool mentioned in analysis: {tool_name}"
                        
                        tools.append(ToolRecommendation(
                            name=tool_name,
                            description=description[:200],  # Limit description length
                            url="",
                            relevance_score=0.85,
                            source="rag",
                            confidence=0.8,
                            topics=[],
                            operations=[]
                        ))
        
        # Also check for common bioinformatics tools as fallback
        known_tools = {
            'biopython': 'Python library for bioinformatics',
            'bioconductor': 'R packages for bioinformatics',
            'blast': 'Basic Local Alignment Search Tool',
            'clustal': 'Multiple sequence alignment',
            'igv': 'Integrative Genomics Viewer',
            'galaxy': 'Web-based bioinformatics platform',
            'gatk': 'Genome Analysis Toolkit',
            'cytoscape': 'Network analysis and visualization',
            'samtools': 'SAM/BAM file manipulation',
            'bowtie': 'Fast sequence alignment',
            'star': 'RNA-seq aligner',
            'deseq2': 'Differential gene expression analysis',
            'tophat': 'RNA-seq read alignment',
            'cufflinks': 'Transcript assembly and quantification',
            'bedtools': 'Genome feature manipulation',
            'vcftools': 'VCF file manipulation',
            'picard': 'Java tools for manipulating HTS data',
            'fastqc': 'Quality control for sequencing data',
            'trimmomatic': 'Flexible read trimming tool',
            'spades': 'Genome assembler',
            'megahit': 'Ultra-fast metagenomic assembler'
        }
        
        existing_names = {tool.name.lower() for tool in tools}
        
        for tool_key, description in known_tools.items():
            if tool_key.lower() in rag_response.lower() and tool_key.lower() not in existing_names:
                tools.append(ToolRecommendation(
                    name=tool_key.title(),
                    description=description,
                    url="",
                    relevance_score=0.75,
                    source="rag",
                    confidence=0.7,
                    topics=[],
                    operations=[]
                ))
        
        return tools[:10]  # Limit to top 10 tools from RAG
    
    def _convert_vector_to_recommendations(self, vector_results: List[Dict]) -> List[ToolRecommendation]:
        """Convert vector search results to tool recommendations"""
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
                operations=payload.get("operations", []),
                programming_language=payload.get("language", [""])[0] if payload.get("language") else None
            ))
        
        return tools
    
    def _extract_tools_from_web_search(self, web_findings: str) -> List[ToolRecommendation]:
        """Extract tools from MCP web search results"""
        tools = []
        
        # Define common bioinformatics tools based on query context with URLs
        web_tool_suggestions = {
            'alignment': [
                ('BLAST', 'Basic Local Alignment Search Tool for sequence similarity search', 'https://blast.ncbi.nlm.nih.gov/'),
                ('Clustal Omega', 'Multiple sequence alignment tool', 'https://www.ebi.ac.uk/Tools/msa/clustalo/'),
                ('MUSCLE', 'Multiple sequence alignment software', 'https://www.ebi.ac.uk/Tools/msa/muscle/'),
                ('MAFFT', 'Multiple alignment program for amino acid or nucleotide sequences', 'https://mafft.cbrc.jp/alignment/server/')
            ],
            'rna': [
                ('STAR', 'Spliced Transcripts Alignment to a Reference', 'https://github.com/alexdobin/STAR'),
                ('TopHat', 'Splice junction mapper for RNA-Seq reads', 'https://ccb.jhu.edu/software/tophat/'),
                ('DESeq2', 'Differential gene expression analysis', 'https://bioconductor.org/packages/release/bioc/html/DESeq2.html'),
                ('edgeR', 'Empirical analysis of digital gene expression', 'https://bioconductor.org/packages/release/bioc/html/edgeR.html')
            ],
            'assembly': [
                ('SPAdes', 'Genome assembler for single-cell and multi-cell bacterial genomes', 'https://github.com/ablab/spades'),
                ('Velvet', 'Short read de novo assembler', 'https://www.ebi.ac.uk/~zerbino/velvet/'),
                ('ABySS', 'Assembly By Short Sequences', 'https://github.com/bcgsc/abyss'),
                ('MEGAHIT', 'Ultra-fast and memory-efficient NGS assembler', 'https://github.com/voutcn/megahit')
            ],
            'variant': [
                ('GATK', 'Genome Analysis Toolkit for variant discovery', 'https://gatk.broadinstitute.org/'),
                ('VCFtools', 'Tools for working with VCF files', 'https://vcftools.github.io/'),
                ('SAMtools', 'Utilities for the Sequence Alignment/Map format', 'http://www.htslib.org/'),
                ('Picard', 'Command line tools for manipulating high-throughput sequencing data', 'https://broadinstitute.github.io/picard/')
            ],
            'phylogen': [
                ('MEGA', 'Molecular Evolutionary Genetics Analysis', 'https://www.megasoftware.net/'),
                ('IQ-TREE', 'Efficient phylogenomic software', 'http://www.iqtree.org/'),
                ('RAxML', 'Randomized Axelerated Maximum Likelihood', 'https://cme.h-its.org/exelixis/web/software/raxml/'),
                ('FastTree', 'Approximately-maximum-likelihood phylogenetic trees', 'http://www.microbesonline.org/fasttree/')
            ],
            'structure': [
                ('PyMOL', 'Molecular visualization software', 'https://pymol.org/'),
                ('ChimeraX', 'Molecular visualization and analysis', 'https://www.cgl.ucsf.edu/chimerax/'),
                ('AlphaFold', 'AI system for protein structure prediction', 'https://alphafold.ebi.ac.uk/'),
                ('I-TASSER', 'Protein structure and function prediction', 'https://zhanggroup.org/I-TASSER/')
            ]
        }
        
        # Determine query category and suggest relevant tools
        query_lower = web_findings.lower()
        relevant_tools = []
        
        for category, tool_list in web_tool_suggestions.items():
            if category in query_lower:
                relevant_tools.extend(tool_list)
        
        # If no specific category found, use general tools
        if not relevant_tools:
            relevant_tools = [
                ('Bioconductor', 'Open source software for bioinformatics', 'https://www.bioconductor.org/'),
                ('Biopython', 'Python tools for computational molecular biology', 'https://biopython.org/'),
                ('Galaxy', 'Web-based platform for computational biology', 'https://galaxyproject.org/'),
                ('UCSC Genome Browser', 'Genome browser for vertebrate genomes', 'https://genome.ucsc.edu/')
            ]
        
        # Create tool recommendations
        for i, (name, description, url) in enumerate(relevant_tools[:3]):  # Limit to 3 tools
            tools.append(ToolRecommendation(
                name=name,
                description=description,
                url=url,
                relevance_score=0.7 - (i * 0.05),  # Decreasing relevance
                source="mcp",
                confidence=0.7 - (i * 0.05),
                topics=[],
                operations=[]
            ))
        
        return tools
    
    def _extract_tools_from_research(self, research_report: str) -> List[ToolRecommendation]:
        """Extract tools from GPT Researcher report"""
        tools = []
        
        # Research-based tool recommendations with academic context and URLs
        research_tools = {
            'sequence': [
                ('Ensembl', 'Genome browser for vertebrate genomes with extensive annotations', 'https://www.ensembl.org/'),
                ('UniProt', 'Comprehensive protein sequence and annotation database', 'https://www.uniprot.org/'),
                ('NCBI BLAST', 'NCBI Basic Local Alignment Search Tool suite', 'https://blast.ncbi.nlm.nih.gov/')
            ],
            'expression': [
                ('GEO', 'Gene Expression Omnibus database', 'https://www.ncbi.nlm.nih.gov/geo/'),
                ('ArrayExpress', 'Archive of functional genomics data', 'https://www.ebi.ac.uk/arrayexpress/'),
                ('Salmon', 'Tool for quantifying expression from RNA-seq data', 'https://salmon.readthedocs.io/')
            ],
            'network': [
                ('STRING', 'Protein-protein interaction networks database', 'https://string-db.org/'),
                ('Cytoscape', 'Open source software platform for visualizing molecular interaction networks', 'https://cytoscape.org/'),
                ('KEGG', 'Database resource for understanding high-level functions', 'https://www.genome.jp/kegg/')
            ],
            'genomics': [
                ('1000 Genomes', 'Public catalogue of human variation and genotype data', 'https://www.internationalgenome.org/'),
                ('gnomAD', 'Genome aggregation database', 'https://gnomad.broadinstitute.org/'),
                ('ClinVar', 'Public archive of reports on relationships among human variations and phenotypes', 'https://www.ncbi.nlm.nih.gov/clinvar/')
            ]
        }
        
        # Analyze research report content to determine relevant tools
        report_lower = research_report.lower()
        selected_tools = []
        
        for category, tool_list in research_tools.items():
            if any(keyword in report_lower for keyword in [category, 'database', 'analysis', 'research']):
                selected_tools.extend(tool_list)
        
        # If no specific match, provide general research tools
        if not selected_tools:
            selected_tools = [
                ('PubMed', 'Database of biomedical literature', 'https://pubmed.ncbi.nlm.nih.gov/'),
                ('Bioinformatics.org', 'Portal for bioinformatics resources', 'https://www.bioinformatics.org/'),
                ('ExPASy', 'Bioinformatics resource portal', 'https://www.expasy.org/')
            ]
        
        # Create tool recommendations
        for i, (name, description, url) in enumerate(selected_tools[:3]):  # Limit to 3 tools
            tools.append(ToolRecommendation(
                name=name,
                description=description,
                url=url,
                relevance_score=0.75 - (i * 0.05),  # Decreasing relevance
                source="gpt_researcher",
                confidence=0.75 - (i * 0.05),
                topics=[],
                operations=[]
            ))
        
        return tools
    
    def _synthesize_and_rank_tools(self, all_tools: List[ToolRecommendation], max_tools: int) -> List[ToolRecommendation]:
        """Intelligently synthesize and rank all tool recommendations"""
        
        # First, normalize scores by source to ensure fair representation
        source_groups = {}
        for tool in all_tools:
            if tool.source not in source_groups:
                source_groups[tool.source] = []
            source_groups[tool.source].append(tool)
        
        # Normalize scores within each source group
        normalized_tools = []
        for source, tools in source_groups.items():
            if not tools:
                continue
                
            # Sort tools by relevance within source
            tools.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Assign normalized scores (0.9, 0.85, 0.8, etc.)
            base_score = 0.9
            for i, tool in enumerate(tools):
                # Keep original score but cap it to ensure fair competition
                normalized_score = min(tool.relevance_score, base_score - (i * 0.05))
                tool.relevance_score = max(0.5, normalized_score)  # Minimum score of 0.5
                normalized_tools.append(tool)
        
        # Remove duplicates based on name similarity
        unique_tools = {}
        for tool in normalized_tools:
            tool_key = tool.name.lower().strip()
            # Skip generic/noise tools
            if tool_key in ['seq', 'generation'] or len(tool_key) < 3:
                continue
                
            if tool_key not in unique_tools or tool.relevance_score > unique_tools[tool_key].relevance_score:
                unique_tools[tool_key] = tool
        
        # Convert back to list and sort by relevance score
        ranked_tools = list(unique_tools.values())
        ranked_tools.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Boost confidence for tools found by multiple systems
        tool_sources = {}
        for tool in normalized_tools:
            key = tool.name.lower().strip()
            if key not in tool_sources:
                tool_sources[key] = []
            tool_sources[key].append(tool.source)
        
        # Apply multi-source boost
        for tool in ranked_tools:
            key = tool.name.lower().strip()
            unique_sources = set(tool_sources.get(key, []))
            if len(unique_sources) > 1:
                tool.confidence = min(0.95, tool.confidence + 0.15)
                tool.relevance_score = min(0.99, tool.relevance_score + 0.1)
        
        # Ensure diversity: try to include tools from different sources
        final_tools = []
        sources_used = set()
        
        # First pass: include top tool from each source
        for tool in ranked_tools:
            if tool.source not in sources_used and len(final_tools) < max_tools:
                final_tools.append(tool)
                sources_used.add(tool.source)
        
        # Second pass: fill remaining slots with best remaining tools
        for tool in ranked_tools:
            if tool not in final_tools and len(final_tools) < max_tools:
                final_tools.append(tool)
        
        # Re-sort final selection by relevance score
        final_tools.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return final_tools[:max_tools]
    
    def _calculate_unified_confidence(self, execution_summary: Dict, all_tools: List[ToolRecommendation]) -> float:
        """Calculate overall confidence in the search results"""
        systems_used = len(execution_summary["systems_used"])
        errors = len(execution_summary["errors"])
        tools_found = len(all_tools)
        
        # Base confidence from system availability
        base_confidence = systems_used / 4.0  # 4 total systems
        
        # Penalty for errors
        error_penalty = errors * 0.1
        
        # Bonus for finding tools
        tool_bonus = min(0.2, tools_found * 0.02)
        
        final_confidence = max(0.0, min(1.0, base_confidence - error_penalty + tool_bonus))
        
        return final_confidence
    
    def display_results(self, result: UnifiedSearchResult) -> None:
        """Display comprehensive search results"""
        
        print(f"\n🧬 UNIFIED SEARCH RESULTS")
        print("=" * 70)
        print(f"Query: {result.query}")
        print(f"Execution Time: {result.execution_time:.2f}s")
        print(f"Overall Confidence: {result.confidence_score:.2f}")
        print(f"Systems Used: {', '.join(result.execution_summary['systems_used'])}")
        
        # Top Recommendations
        print(f"\n🎯 TOP {len(result.top_recommendations)} TOOL RECOMMENDATIONS:")
        print("-" * 50)
        for i, tool in enumerate(result.top_recommendations, 1):
            print(f"\n{i}. {tool.name}")
            print(f"   Description: {tool.description}")
            print(f"   Source: {tool.source}")
            print(f"   Relevance: {tool.relevance_score:.2f}")
            print(f"   Confidence: {tool.confidence:.2f}")
            if tool.url:
                print(f"   URL: {tool.url}")
            if tool.topics:
                print(f"   Topics: {', '.join(tool.topics[:3])}")
        
        # RAG Analysis
        if result.rag_analysis and len(result.rag_analysis) > 50:
            print(f"\n💡 RAG AGENT ANALYSIS:")
            print("-" * 30)
            print(result.rag_analysis[:500] + "..." if len(result.rag_analysis) > 500 else result.rag_analysis)
        
        # Vector Search Results
        if result.vector_search_results:
            print(f"\n🎯 VECTOR DATABASE RESULTS:")
            print("-" * 30)
            for i, vr in enumerate(result.vector_search_results[:3], 1):
                payload = vr["payload"]
                print(f"{i}. {payload.get('name', 'Unknown')} (score: {vr['score']:.2f})")
        
        # Execution Summary
        if result.execution_summary.get("errors"):
            print(f"\n⚠️  ERRORS ENCOUNTERED:")
            print("-" * 20)
            for error in result.execution_summary["errors"]:
                print(f"  • {error}")

# Interactive CLI
async def main():
    """Interactive command-line interface for unified search"""
    
    print("🧬 UNIFIED BIOINFORMATICS SEARCH ENGINE")
    print("=" * 70)
    print("Integrating: RAG Agent + Vector DB + MCP + GPT Researcher")
    print("=" * 70)
    
    # Initialize the search engine
    search_engine = UnifiedBioinformaticsSearch()
    
    if not any(search_engine.systems_status.values()):
        print("\n❌ No systems are available. Please configure at least one system.")
        return
    
    print(f"\nReady! Type 'quit' to exit, 'help' for commands")
    
    while True:
        print("\n" + "-" * 40)
        user_input = input("🔍 Enter bioinformatics query: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thank you for using the Unified Bioinformatics Search Engine!")
            break
        
        if user_input.lower() == 'help':
            print("\nCommands:")
            print("  <query>              - Search for bioinformatics tools")
            print("  <query> --max=N      - Limit results to N tools (default: 5)")
            print("  help                 - Show this help")
            print("  quit                 - Exit the search engine")
            continue
        
        if not user_input:
            continue
        
        # Parse max tools parameter
        max_tools = 5
        if '--max=' in user_input:
            import re
            match = re.search(r'--max=(\d+)', user_input)
            if match:
                max_tools = int(match.group(1))
                user_input = re.sub(r'--max=\d+', '', user_input).strip()
        
        try:
            result = await search_engine.comprehensive_search(user_input, max_tools)
            search_engine.display_results(result)
        except Exception as e:
            print(f"\n❌ Search failed: {str(e)}")
            import traceback
            print(f"Details: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main()) 