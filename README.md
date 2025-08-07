# Resume JD Matcher

An AI-powered system that matches resumes to job descriptions using advanced LLMs (Groq/OpenAI), Google embeddings, and asynchronous processing pipeline.

## Features

- **Resume Parsing**: Extracts structured data from PDF resumes using LlamaCloud with async support
- **Vector Search**: Uses ChromaDB with Google embeddings for semantic matching and duplicate detection
- **Intelligent Ranking**: Multi-factor ranking using vector similarity and LLM analysis
- **Email Notifications**: Configurable email notifications with PDF attachments for recruiters and requestors
- **Result Persistence**: MongoDB integration for storing job search results
- **Async Pipeline**: Fully asynchronous processing for better performance

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

3. Create a `.env` file with your API keys and configuration:
```
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
LLAMA_CLOUD_API_KEY=your_llama_cloud_key
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_app_password
```

4. Set up MongoDB for result persistence.

## Usage

1. Place resume PDFs in the `data/` directory
2. Use either:
   - Jupyter notebooks in `notebook/` for interactive use
   - Import `pipeline.py` for programmatic use:
   ```python
   from pipeline import run_pipeline
   
   await run_pipeline(
       resumes_dir_path="./data",
       jd="your job description",
       emails=EmailConfig(
           ar_requestor="requestor@company.com",
           recruiter="recruiter@company.com"
       )
   )
   ```

## Architecture

The system uses a 4-stage async pipeline:

1. **Extract & Store**: 
   - Parses resumes using LlamaCloud
   - Stores in persistent ChromaDB with duplicate detection
   - Generates Google embeddings for matching

2. **Comparison**:
   - Vector similarity search for initial matches
   - Returns top 3 candidates with similarity scores

3. **Ranking**:
   - LLM-based detailed analysis
   - Considers skills, experience, and vector similarity
   - Updates match scores using weighted criteria

4. **Communication**:
   - Stores results in MongoDB
   - Sends detailed email reports
   - Attaches top candidate PDFs

## Requirements

- Python 3.8+
- MongoDB
- SMTP server access
- API keys for:
  - Groq/OpenAI (LLM)
  - Google (Embeddings)
  - LlamaCloud (PDF extraction)