import os
import asyncio
from typing import List, TypedDict, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LangChain / Llama-related imports
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

from llama_cloud_services import LlamaExtract
from llama_cloud_services.extract import SourceText

# ---------------------------------------------------------------------------
#  Environment & model setup – executed once on import
# ---------------------------------------------------------------------------
load_dotenv()

os.environ["GROQ_API_KEY"]   = os.getenv("GROQ_API_KEY", "")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv("LLAMA_CLOUD_API_KEY", "")

# Ensure a running event loop is available for grpc.aio (used by GoogleGenerativeAIEmbeddings)
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

llm_groq = ChatGroq(model="deepseek-r1-distill-llama-70b", max_tokens=2000, temperature=0)
llm_openai = ChatOpenAI(model="gpt-4o", max_tokens=2000, temperature=0)

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Single persistent Chroma collection for the whole app
vectorstore = Chroma(
    collection_name="resume_collection",
    embedding_function=embedding_model,
    persist_directory="./chromaDB",
)

# ---------------------------------------------------------------------------
#  Pydantic models & typed state helpers
# ---------------------------------------------------------------------------
class EmailConfig(BaseModel):
    """Configuration for who should receive which email."""

    ar_requestor: str
    recruiter: str


class Resume(BaseModel):
    name: str = Field(description="Full name of candidate")
    email: str = Field(description="Email address")
    skills: list[str] = Field(description="Technical skills and technologies")
    work_experience: list[str] = Field(description="Work experience details")
    education: str = Field(description="Educational background")
    projects: list[str] = Field(description="Projects worked on")
    rank_reason: Optional[str] = Field(
        default=None,
        description="Specific reason for ranking this resume higher"
    )


class RankedResumes(BaseModel):
    ranked_resumes: List[Resume] = Field(
        description="List of resumes ranked by relevance to the job description"
    )


class State(TypedDict):
    resumes_dir_path: str
    jd: str
    text_resumes: List[str]
    matched_resumes: List
    ranked_resumes: List
    no_match: bool
    emails: Optional[EmailConfig]
    resume_file_path_mapping: dict


# ---------------------------------------------------------------------------
#  Llama-index extractors & structured LLM helpers
# ---------------------------------------------------------------------------
extractor = LlamaExtract()

# Check if agents exist, create only if they don't
def get_or_create_agent(extractor, name: str, data_schema):
    """Get existing agent by name or create a new one if it doesn't exist."""
    try:
        # Try to get existing agent by name
        agent = extractor.get_agent(name=name)
        print(f"Using existing agent: {name}")
        return agent
    except Exception:
        # Agent doesn't exist, create a new one
        print(f"Creating new agent: {name}")
        return extractor.create_agent(name=name, data_schema=data_schema)

agent_resume = get_or_create_agent(extractor, "resume-parser-pipeline", Resume)
agent_ranked = get_or_create_agent(extractor, "ranked-resumes-pipeline", RankedResumes)

str_llm_groq = llm_groq.with_structured_output(RankedResumes)
str_llm_openai = llm_openai.with_structured_output(RankedResumes)

# ---------------------------------------------------------------------------
#  Helper functions reused across the pipeline
# ---------------------------------------------------------------------------

def resume_to_text(resume: dict) -> str:
    """Convert a single extracted resume json -> plain text for embeddings."""
    text = f"{resume['name']} is a skilled professional.\n"
    text += f"Email: {resume['email']}\n"
    text += f"Skills: {', '.join(resume['skills'])}\n\n"

    text += "Work Experience:\n"
    for exp in resume["work_experience"]:
        text += f"- {exp}\n"

    text += f"\nEducation: {resume['education']}\n\nProjects:\n"
    for proj in resume["projects"]:
        text += f"- {proj}\n"
    return text.strip()


def clear_vector_store():
    """Clear all documents from the vector store. Use with caution!"""
    global vectorstore
    try:
        # Delete the collection and recreate it
        vectorstore.delete_collection()
        print("Vector store cleared successfully")
        
        # Recreate the collection
        vectorstore = Chroma(
            collection_name="resume_collection",
            embedding_function=embedding_model,
            persist_directory="./chromaDB",
        )
        print("Vector store recreated")
    except Exception as e:
        print(f"Error clearing vector store: {e}")


def get_vector_store_stats():
    """Get statistics about the current vector store"""
    try:
        existing_docs = vectorstore.get()
        if existing_docs and existing_docs.get('metadatas'):
            emails = [meta.get('email', 'Unknown') for meta in existing_docs['metadatas']]
            names = [meta.get('name', 'Unknown') for meta in existing_docs['metadatas']]
            return {
                "total_documents": len(emails),
                "unique_names": list(set(names)),
                "duplicate_count": len(emails) - len(set(emails))
            }
        return {"total_documents": 0, "unique_names": [], "duplicate_count": 0}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
#  Pipeline nodes (copied from notebook, trimmed of print/debug)
# ---------------------------------------------------------------------------
async def extract_and_store_resumes(state: State):
    """Extract text from PDFs in *state['resumes_dir_path']* and persist to Chroma."""

    resumes_dir_path = state["resumes_dir_path"]
    pdf_files = [
        os.path.join(resumes_dir_path, f)
        for f in os.listdir(resumes_dir_path)
        if f.lower().endswith(".pdf")
    ]

    jobs = await agent_resume.queue_extraction(pdf_files)

    job_to_file_mapping = {job.id: pdf_files[i] for i, job in enumerate(jobs)}

    # Wait for all jobs
    results = []
    for job in jobs:
        while True:
            result = agent_resume.get_extraction_run_for_job(job.id)
            status = result.status.value
            if status in {"COMPLETED", "SUCCESS"}:
                results.append(result)
                break
            elif status == "FAILED":
                break
            await asyncio.sleep(2)

    resumes, resume_file_mapping = [], {}
    for result in results:
        if not result.data:
            continue
        file_path = job_to_file_mapping.get(result.job_id)
        resume_info = {
            "name": result.data["name"],
            "email": result.data["email"],
            "skills": result.data["skills"],
            "work_experience": result.data["work_experience"],
            "education": result.data["education"],
            "projects": result.data["projects"],
        }
        resumes.append(resume_info)
        resume_file_mapping[resume_info["name"]] = file_path

    # Check for existing resumes to prevent duplicates (using email as unique identifier)
    existing_emails = set()
    try:
        # Get all existing documents from the vector store
        existing_docs = vectorstore.get()
        if existing_docs and existing_docs.get('metadatas'):
            existing_emails = {meta.get('email') for meta in existing_docs['metadatas'] if meta.get('email')}
    except Exception as e:
        print(f"Warning: Could not check existing documents: {e}")

    # Convert to plain-text docs for embeddings, filtering out duplicates
    docs = []
    new_resumes = []
    for resume in resumes:
        if resume["email"] not in existing_emails:
            text = resume_to_text(resume)
            doc = Document(
                page_content=text,
                metadata={
                    "name": resume["name"],
                    "email": resume["email"],
                    "skills": ", ".join(s.lower().strip() for s in resume["skills"]),
                },
            )
            docs.append(doc)
            new_resumes.append(resume)
            print(f"Adding new resume: {resume['name']} ({resume['email']})")
        else:
            print(f"Skipping duplicate resume: {resume['name']} ({resume['email']}) - email already exists")

    if docs:
        vectorstore.add_documents(docs)
        vectorstore.persist()
        print(f"Added {len(docs)} new resumes to vector store")
    else:
        print("No new resumes to add - all were duplicates")

    return {
        "text_resumes": [resume_to_text(r) for r in resumes],
        "resume_file_path_mapping": resume_file_mapping,
    }


async def comparison_agent(state: State):
    """Vector-similarity search to get top 3 matches for the JD."""
    jd_text = state["jd"]
    matched = vectorstore.similarity_search(jd_text, k=3)
    return {"matched_resumes": matched}


async def ranking_agent(state: State):
    """Rank the matched resumes via LLM for extra relevance ordering."""
    matched_resumes = state["matched_resumes"]
    if not matched_resumes:
        return {"ranked_resumes": [], "no_match": True}

    # Build prompt
    resume_texts = []
    for doc in matched_resumes:
        resume_texts.append(
            {
                "name": doc.metadata.get("name", "Unknown"),
                "email": doc.metadata.get("email", "Unknown"),
                "content": doc.page_content,
                "skills": doc.metadata.get("skills", ""),
            }
        )

    ranking_prompt = f"""
Job Description: {state['jd']}

Please rank the following resumes from most relevant (1) to least relevant based on the job description.
Return only the ranking as a numbered list with name and email. Also include skills & work experience. and a clear reason for the ranking for each resume.
"""
    for i, resume in enumerate(resume_texts, 1):
        ranking_prompt += (
            f"\n{i}. Name: {resume['name']}\n   Email: {resume['email']}\n   Skills: {resume['skills']}\n   Content: {resume['content'][:400]}...\n"
        )

    ranking_response = llm_groq.invoke(
        [
            SystemMessage(
                content="You are a recruitment expert. Rank the resumes based on relevance to the job description."
            ),
            HumanMessage(content=ranking_prompt),
        ]
    )

    ranked = agent_ranked.extract(SourceText(text_content=ranking_response.content))
    return {
        "ranked_resumes": ranked.data["ranked_resumes"],
        "no_match": False,
    }


# -----------------------------
#  Minimal communicator (email)
# -----------------------------
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication


async def communicator_agent(state: State):
    """Send summary email with the top ranked resumes (or no-match notice)."""

    emails_cfg: EmailConfig | None = state.get("emails")
    if not emails_cfg:
        return  # Email sending disabled

    no_match = state["no_match"]
    ranked_resumes = state.get("ranked_resumes", [])
    resume_file_path_mapping = state.get("resume_file_path_mapping", {})

    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")

    if not sender_email or not sender_password:
        return

    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(sender_email, sender_password)

    msg = MIMEMultipart()
    msg["From"] = sender_email

    if no_match:
        msg["To"] = emails_cfg.recruiter
        msg["Subject"] = "No Matching Resumes Found"
        msg.attach(
            MIMEText(
                "Dear Recruiter,\n\nWe could not find any resumes matching the job description.\n\nBest regards,\nResume Matching System",
                "plain",
            )
        )
    else:
        msg["To"] = emails_cfg.ar_requestor
        msg["Subject"] = "Ranked Resumes for Job Description"

        top_resumes = ranked_resumes[:3]
        body_parts = []
        for i, cand in enumerate(top_resumes, 1):
            part = (
                f"{i}. Name: {cand['name']}\n   Email: {cand['email']}\n   Skills: {', '.join(cand['skills'])}\n   Work Experience:\n     - "
                + "\n     - ".join(cand["work_experience"])
            )
            # Add rank reason if available
            if cand.get('rank_reason'):
                part += f"\n   Ranking Reason: {cand['rank_reason']}"
            part += "\n\n"
            body_parts.append(part)
        msg.attach(
            MIMEText(
                "Dear AR Requestor,\n\nHere are the top candidates:\n\n" + "\n".join(body_parts),
                "plain",
            )
        )

        # Attach PDF copies
        for cand in top_resumes:
            path = resume_file_path_mapping.get(cand["name"])
            if not path or not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                attachment = MIMEApplication(f.read(), _subtype="pdf")
                attachment.add_header(
                    "Content-Disposition", "attachment", filename=f"{cand['name']}_resume.pdf"
                )
                msg.attach(attachment)

    server.sendmail(sender_email, msg["To"], msg.as_string())
    server.quit()


# ---------------------------------------------------------------------------
#  Public helper to run the whole workflow sequentially
# ---------------------------------------------------------------------------
async def run_pipeline(resumes_dir_path: str, jd: str, emails: EmailConfig | None = None, clear_after_processing: bool = False):
    """Convenience wrapper that executes all steps and returns the final state.
    
    Args:
        resumes_dir_path: Path to directory containing resume PDFs
        jd: Job description text
        emails: Email configuration for notifications
        clear_after_processing: If True, clears the vector store after successful processing
    """

    state: State = {
        "resumes_dir_path": resumes_dir_path,
        "jd": jd,
        "emails": emails,
    }

    state.update(await extract_and_store_resumes(state))
    state.update(await comparison_agent(state))
    state.update(await ranking_agent(state))
    await communicator_agent(state)
    
    # Optionally clear vector store after successful processing
    if clear_after_processing:
        print("Clearing vector store after successful processing...")
        clear_vector_store()
    
    return state 