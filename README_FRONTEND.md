# BioTools Search - Perplexity-Style Frontend

A beautiful, modern web interface for your unified bioinformatics search engine, inspired by Perplexity AI's design.

## Features

### 🎨 Modern Design
- **Dark theme** with beautiful gradients and animations
- **Responsive design** that works on desktop, tablet, and mobile
- **Perplexity-inspired UI** with clean, professional aesthetics
- **Real-time system status** indicator in the header

### 🚀 Advanced Search Experience
- **Intelligent search box** with autocomplete suggestions
- **Example queries** organized by bioinformatics categories
- **Real-time loading states** with step-by-step progress
- **Multi-system search** across RAG, Vector DB, MCP, and GPT Researcher

### 📊 Rich Results Display
- **Confidence scoring** with color-coded indicators
- **Tool cards** with relevance scores and metadata
- **Source attribution** showing which system found each tool
- **Interactive elements** like copy-to-clipboard functionality
- **Detailed analysis** sections when available

### 🔧 System Integration
- **Live status monitoring** of all search systems
- **Error handling** with user-friendly messages
- **Session management** for search history
- **Performance metrics** display

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Ensure Environment Setup
Make sure your `.env` file contains all required API keys:
```bash
# Required for all systems
GOOGLE_API_KEY=your_google_api_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key

# Optional for enhanced features
EXA_API_KEY=your_exa_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 3. Launch the Web Application
```bash
python web_app.py
```

The application will start on `http://localhost:5000`

## Usage Guide

### Basic Search
1. **Enter a query** in the search box (e.g., "protein sequence alignment tools")
2. **Press Enter** or click the search button
3. **Watch the progress** as systems are queried in real-time
4. **Review results** with confidence scores and source attribution

### Example Categories
- **Sequence Analysis**: Alignment, annotation, and comparison tools
- **Genomics**: RNA-seq, variant calling, and expression analysis
- **Assembly**: Genome and transcriptome assembly software
- **Phylogenetics**: Evolutionary tree construction and analysis
- **Variant Analysis**: SNP calling and genome variation tools
- **Structural Biology**: Protein structure prediction and analysis

### Understanding Results

#### Confidence Scores
- 🟢 **High (80%+)**: Very reliable recommendations
- 🟡 **Medium (60-79%)**: Good recommendations with some uncertainty
- 🔴 **Low (<60%)**: Suggestions that may need verification

#### Source Systems
- 🧠 **RAG**: AI-powered analysis with contextual understanding
- 🗃️ **Vector DB**: Semantic similarity search through tool database
- 🌐 **MCP**: Web search validation and current information
- 📄 **GPT Researcher**: Comprehensive research report generation

#### Tool Information
Each tool card displays:
- **Name and description** of the tool
- **Relevance score** based on query match
- **Confidence level** from the AI analysis
- **Topic tags** for easy categorization
- **Direct links** to tool websites (when available)
- **Copy functionality** for easy sharing

## API Endpoints

### GET /api/status
Returns the current status of all search systems.

**Response:**
```json
{
  "status": "ready",
  "systems": {
    "rag": true,
    "vector_db": true,
    "mcp": true,
    "gpt_researcher": true
  },
  "active_systems": 4,
  "total_systems": 4
}
```

### POST /api/search
Execute a search query across all systems.

**Request:**
```json
{
  "query": "protein sequence alignment tools",
  "max_tools": 5
}
```

**Response:**
```json
{
  "session_id": "uuid",
  "query": "protein sequence alignment tools",
  "execution_time": 12.34,
  "confidence_score": 0.89,
  "systems_used": ["rag", "vector_db", "mcp"],
  "total_tools_found": 5,
  "recommendations": [...],
  "rag_analysis": "Detailed analysis...",
  "errors": []
}
```

### GET /api/examples
Get example queries organized by category.

## Customization

### Styling
The interface uses CSS custom properties for easy theming:
```css
:root {
  --bg-primary: #0f1419;      /* Main background */
  --bg-secondary: #1a1f2e;    /* Card backgrounds */
  --accent-primary: #3b82f6;  /* Primary blue */
  --accent-secondary: #10b981; /* Success green */
  /* ... more variables */
}
```

### Adding New Example Queries
Edit the `examples()` function in `web_app.py`:
```python
examples = [
    {
        'query': 'your search query',
        'description': 'What this search finds',
        'category': 'Your Category'
    },
    # ... more examples
]
```

## Architecture

```
Frontend (HTML/CSS/JS)
    ↓
Flask Web Server (web_app.py)
    ↓
Unified Search Engine (unified_bioinformatics_search.py)
    ↓
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ RAG Agent   │ Vector DB   │ MCP System  │ GPT Research│
└─────────────┴─────────────┴─────────────┴─────────────┘
```

## Performance

- **Average search time**: 8-15 seconds for comprehensive search
- **Concurrent handling**: Multiple searches supported
- **Timeout protection**: 2-minute maximum search time
- **Resource optimization**: Efficient system utilization

## Troubleshooting

### Common Issues

1. **Search engine not available**
   - Check that all environment variables are set
   - Verify Qdrant and other services are accessible
   - Restart the application

2. **Slow searches**
   - Check network connectivity
   - Verify API key limits aren't exceeded
   - Consider reducing max_tools parameter

3. **Missing results**
   - Verify individual system configurations
   - Check the system status endpoint
   - Review error logs for specific issues

### Development

For development with auto-reload:
```bash
export FLASK_ENV=development
python web_app.py
```

For production deployment:
```bash
gunicorn --bind 0.0.0.0:5000 web_app:app
```

## Contributing

To add new features:
1. Update the backend logic in `web_app.py`
2. Modify the frontend in `templates/index.html`
3. Test across different devices and browsers
4. Update this documentation

---

**Built with ❤️ for the bioinformatics community** 