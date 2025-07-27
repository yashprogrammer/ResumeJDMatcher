# Resume JD Matcher

An AI-powered system that matches resumes to job descriptions using LangGraph, vector embeddings, and LLM-based ranking.

## Features

- **Resume Parsing**: Extracts structured data from PDF resumes using LlamaCloud
- **Vector Search**: Uses ChromaDB with Google embeddings for semantic matching
- **Intelligent Ranking**: Ranks candidates based on JD relevance using LLMs
- **Workflow Orchestration**: Built with LangGraph for clear processing pipeline

## Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/ResumeJDMatcher.git
cd ResumeJDMatcher
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
LLAMA_CLOUD_API_KEY=your_llama_cloud_key
```

4. Run the Jupyter notebook:
```bash
jupyter notebook notebook/POC_langGraph.ipynb
```

## Usage

1. Place resume PDFs in the `data/` directory
2. Run the workflow with your job description
3. Get ranked candidates based on relevance

## Architecture

The system uses a 3-stage LangGraph workflow:
- **Extract & Store**: Parse resumes and store in vector database
- **Comparison**: Find matching resumes using semantic search
- **Ranking**: Rank candidates using LLM analysis

## Requirements

- Python 3.8+
- API keys for Groq, OpenAI, Google, and LlamaCloud