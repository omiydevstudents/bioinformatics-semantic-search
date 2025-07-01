#!/usr/bin/env python3
"""
Perplexity-style Web Frontend for Unified Bioinformatics Search
"""

from flask import Flask, render_template, request, jsonify, stream_template
from flask_cors import CORS
import asyncio
import json
import time
from datetime import datetime
import threading
from unified_bioinformatics_search import UnifiedBioinformaticsSearch
import uuid

app = Flask(__name__)
CORS(app)

# Global search engine instance
search_engine = None
search_sessions = {}  # Store search sessions

def initialize_search_engine():
    """Initialize the search engine in a separate thread"""
    global search_engine
    try:
        search_engine = UnifiedBioinformaticsSearch()
        print("✅ Search engine initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize search engine: {e}")
        search_engine = None

# Initialize on startup
initialize_search_engine()

@app.route('/')
def index():
    """Main page with Perplexity-style interface"""
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    """Execute search and return results"""
    if not search_engine:
        return jsonify({
            'error': 'Search engine not available',
            'message': 'Please check your configuration and try again'
        }), 500
    
    data = request.get_json()
    query = data.get('query', '').strip()
    max_tools = data.get('max_tools', 5)
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    # Generate session ID
    session_id = str(uuid.uuid4())
    
    try:
        # Run async search in thread
        def run_search():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    search_engine.comprehensive_search(query, max_tools)
                )
                search_sessions[session_id] = result
            except Exception as e:
                search_sessions[session_id] = {'error': str(e)}
            finally:
                loop.close()
        
        thread = threading.Thread(target=run_search)
        thread.start()
        thread.join(timeout=120)  # 2 minute timeout
        
        if session_id not in search_sessions:
            return jsonify({'error': 'Search timed out'}), 408
        
        result = search_sessions[session_id]
        
        # Check if result is an error dictionary (more robust checking)
        if isinstance(result, dict) and 'error' in result:
            return jsonify({'error': result['error']}), 500
        
        # Debug logging to see what we got
        print(f"DEBUG: Result type: {type(result)}")
        print(f"DEBUG: Result attributes: {dir(result)}")
        
        # Convert result to JSON-serializable format with robust error handling
        try:
            # Safely extract basic information
            query = getattr(result, 'query', '')
            execution_time = getattr(result, 'execution_time', 0)
            confidence_score = getattr(result, 'confidence_score', 0)
            
            # Safely extract execution summary
            execution_summary = getattr(result, 'execution_summary', {})
            systems_used = execution_summary.get('systems_used', []) if isinstance(execution_summary, dict) else []
            
            # Safely extract recommendations
            top_recommendations = getattr(result, 'top_recommendations', [])
            recommendations_list = []
            
            for tool in top_recommendations:
                try:
                    tool_dict = {
                        'name': getattr(tool, 'name', 'Unknown Tool'),
                        'description': getattr(tool, 'description', ''),
                        'url': getattr(tool, 'url', ''),
                        'relevance_score': getattr(tool, 'relevance_score', 0),
                        'confidence': getattr(tool, 'confidence', 0),
                        'source': getattr(tool, 'source', ''),
                        'topics': getattr(tool, 'topics', []) or [],
                        'operations': getattr(tool, 'operations', []) or [],
                        'programming_language': getattr(tool, 'programming_language', '') or ''
                    }
                    recommendations_list.append(tool_dict)
                except Exception as tool_error:
                    print(f"Error processing tool: {tool_error}")
                    continue
            
            # Safely extract additional data
            rag_analysis = getattr(result, 'rag_analysis', '')
            vector_search_results = getattr(result, 'vector_search_results', [])
            web_search_findings = getattr(result, 'web_search_findings', '')
            research_report = getattr(result, 'research_report', '')
            
            response_data = {
                'session_id': session_id,
                'query': str(query),
                'execution_time': float(execution_time),
                'confidence_score': float(confidence_score),
                'systems_used': systems_used,
                'total_tools_found': len(recommendations_list),
                'recommendations': recommendations_list,
                'rag_analysis': str(rag_analysis) if rag_analysis and len(str(rag_analysis)) > 50 else '',
                'vector_results': vector_search_results[:3] if isinstance(vector_search_results, list) else [],
                'web_findings': str(web_search_findings) if web_search_findings else '',
                'research_report': str(research_report) if research_report else '',
                'errors': execution_summary.get('errors', []) if isinstance(execution_summary, dict) else []
            }
            
            return jsonify(response_data)
        
        except Exception as processing_error:
            print(f"Error processing search result: {processing_error}")
            return jsonify({'error': f'Result processing failed: {str(processing_error)}'}), 500
        
    except Exception as e:
        return jsonify({'error': f'Search failed: {str(e)}'}), 500

@app.route('/api/status')
def status():
    """Get system status"""
    if not search_engine:
        return jsonify({
            'status': 'error',
            'message': 'Search engine not initialized',
            'systems': {}
        })
    
    return jsonify({
        'status': 'ready',
        'systems': search_engine.systems_status,
        'active_systems': sum(search_engine.systems_status.values()),
        'total_systems': len(search_engine.systems_status)
    })

@app.route('/api/examples')
def examples():
    """Get example queries"""
    examples = [
        {
            'query': 'protein sequence alignment tools',
            'description': 'Find tools for aligning protein sequences',
            'category': 'Sequence Analysis'
        },
        {
            'query': 'RNA-seq differential expression analysis',
            'description': 'Tools for analyzing gene expression differences',
            'category': 'Genomics'
        },
        {
            'query': 'genome assembly software',
            'description': 'Tools for assembling genomic sequences',
            'category': 'Assembly'
        },
        {
            'query': 'phylogenetic tree construction',
            'description': 'Software for building evolutionary trees',
            'category': 'Phylogenetics'
        },
        {
            'query': 'variant calling pipeline',
            'description': 'Tools for identifying genetic variants',
            'category': 'Variant Analysis'
        },
        {
            'query': 'protein structure prediction',
            'description': 'Software for predicting 3D protein structures',
            'category': 'Structural Biology'
        }
    ]
    
    return jsonify(examples)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 