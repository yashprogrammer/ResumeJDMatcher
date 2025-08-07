from __future__ import annotations

import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from pymongo import MongoClient
import gridfs

# ---------------------------------------------------------------------------
# MongoDB connection helpers
# ---------------------------------------------------------------------------

def _get_mongo_client() -> MongoClient:
    """Return a cached MongoDB client instance."""
    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    return MongoClient(mongo_uri)


def _get_db():
    """Return handle to the configured MongoDB database."""
    db_name = os.getenv("MONGODB_DB", "resume_jd_matcher")
    return _get_mongo_client()[db_name]


# ---------------------------------------------------------------------------
# Public CRUD helpers
# ---------------------------------------------------------------------------

def insert_job_result(
    jd: str,
    ranked_resumes: List[Dict[str, Any]],
    resume_file_path_mapping: Dict[str, str] | None = None,
) -> str:
    """Persist the JD comparison result to MongoDB.

    Each ranked resume is optionally linked to its PDF file stored in GridFS.

    Args:
        jd: The job-description text that was matched.
        ranked_resumes: List of ranked resume dictionaries (output of the pipeline).
        resume_file_path_mapping: Mapping from *candidate name* -> *local PDF path*.

    Returns:
        The inserted MongoDB document ID as a hex-string.
    """

    if resume_file_path_mapping is None:
        resume_file_path_mapping = {}

    db = _get_db()
    fs = gridfs.GridFS(db)

    enriched_resumes: List[Dict[str, Any]] = []
    for cand in ranked_resumes:
        # Attach the corresponding PDF to GridFS (if we have a local path)
        file_path = resume_file_path_mapping.get(cand["name"])
        pdf_file_id: Optional[Any] = None
        if file_path and os.path.exists(file_path):
            with open(file_path, "rb") as f:
                pdf_file_id = fs.put(f.read(), filename=os.path.basename(file_path))

        # Copy candidate dict and attach the GridFS id (if any)
        cand_doc = cand.copy()
        cand_doc["pdf_file_id"] = pdf_file_id  # May be None if PDF not found / uploaded
        enriched_resumes.append(cand_doc)

    doc = {
        "jd": jd,
        "ranked_resumes": enriched_resumes,
        "created_at": datetime.utcnow(),
    }

    result = db["jd_results"].insert_one(doc)
    return str(result.inserted_id)


def fetch_all_results() -> List[Dict[str, Any]]:
    """Return all previously stored JD ↔ resume comparison results (latest first)."""
    db = _get_db()
    return list(db["jd_results"].find().sort("created_at", -1))


def get_pdf_bytes(file_id) -> Optional[bytes]:
    """Return the raw PDF bytes stored in GridFS by *file_id* (or None on failure)."""
    if not file_id:
        return None
    db = _get_db()
    fs = gridfs.GridFS(db)
    try:
        return fs.get(file_id).read()
    except Exception:
        # Could be file not found / wrong ID
        return None
