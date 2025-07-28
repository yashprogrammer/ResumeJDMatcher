import os
import tempfile
import asyncio

import streamlit as st

from pipeline import (
    EmailConfig,
    extract_and_store_resumes,
    comparison_agent,
    ranking_agent,
    communicator_agent,
    clear_vector_store,
    get_vector_store_stats,
)

st.set_page_config(page_title="Resume ↔ JD Matcher", page_icon="📄")

st.title("📄 Resume ↔ JD Matcher")

# ------------------------------------------------------------
# Sidebar – configuration inputs
# ------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    ar_requestor_email = st.text_input("AR Requestor Email")
    recruiter_email = st.text_input("Recruiter Email")

    st.markdown("---")
    st.subheader("Vector Store Management")
    
    # Show vector store statistics
    stats = get_vector_store_stats()
    if "error" not in stats:
        st.write(f"📊 Total Documents: {stats['total_documents']}")
        st.write(f"👥 Unique Resumes: {len(stats['unique_names'])}")
        if stats['duplicate_count'] > 0:
            st.warning(f"⚠️ Duplicates Found: {stats['duplicate_count']}")
        
        if stats['unique_names']:
            with st.expander("View Stored Resumes"):
                for name in stats['unique_names']:
                    st.write(f"• {name}")
    
    # Clear vector store button
    if st.button("🗑️ Clear Vector Store", help="Remove all stored resumes"):
        clear_vector_store()
        st.success("Vector store cleared!")
        st.rerun()

    st.markdown("---")
    st.subheader("Environment Keys")
    st.caption("Keys are pulled from your local environment. ✅ means detected.")
    for key in [
        "GROQ_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "LLAMA_CLOUD_API_KEY",
        "SENDER_EMAIL",
        "SENDER_PASSWORD",
    ]:
        present = "✅" if os.getenv(key) else "⚠️"
        st.write(f"{present} `{key}`")

# ------------------------------------------------------------
# Main input area
# ------------------------------------------------------------
jd_text = st.text_area("Job Description", height=180)

uploaded_files = st.file_uploader(
    "Upload Resume PDFs", type=["pdf"], accept_multiple_files=True
)

run_button = st.button("🚀 Run Comparison", disabled=not (jd_text and uploaded_files))

status_placeholder = st.empty()
progress_bar = st.progress(0.0)

if run_button:
    if not (ar_requestor_email and recruiter_email):
        st.error("Please fill in both email addresses in the sidebar.")
        st.stop()

    # Save uploaded PDFs to a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        for file in uploaded_files:
            file_path = os.path.join(tmpdir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

        email_cfg = EmailConfig(
            ar_requestor=ar_requestor_email, recruiter=recruiter_email
        )

        # --------------------------------------------------------
        # We execute each node manually so we can update progress
        # --------------------------------------------------------
        state = {
            "resumes_dir_path": tmpdir,
            "jd": jd_text,
            "emails": email_cfg,
        }

        async def workflow():
            # 1️⃣ Extract & store resumes
            status_placeholder.info("Extracting resumes & building vector index …")
            state.update(await extract_and_store_resumes(state))
            progress_bar.progress(0.33)

            # 2️⃣ Compare JD → vector search
            status_placeholder.info("Comparing JD with resumes …")
            state.update(await comparison_agent(state))

            # 3️⃣ Rank via LLM
            status_placeholder.info("Ranking profiles …")
            state.update(await ranking_agent(state))
            progress_bar.progress(0.66)

            # 4️⃣ Send email
            status_placeholder.info("Sending email to AR Requestor …")
            await communicator_agent(state)
            progress_bar.progress(1.0)

            status_placeholder.success("✅ Workflow completed!")
            return state

        # Use a persistent asyncio event loop across Streamlit reruns to avoid
        # "Event loop is closed" errors that occur when the loop is closed while
        # background tasks (e.g., httpx connection cleanup) are still pending.
        if "async_loop" not in st.session_state:
            st.session_state["async_loop"] = asyncio.new_event_loop()

        loop = st.session_state["async_loop"]

        # Make sure our loop is the current one for this thread.
        asyncio.set_event_loop(loop)

        # Run the workflow synchronously inside the persistent loop.
        final_state = loop.run_until_complete(workflow())

        # --------------------------------------------------------
        # Display results
        # --------------------------------------------------------
        st.subheader("Top 3 Matches")
        if final_state.get("ranked_resumes"):
            for i, cand in enumerate(final_state["ranked_resumes"][:3], 1):
                st.markdown(
                    f"**{i}. {cand['name']}**  \n"
                    f":e-mail: {cand['email']}  \n"
                    f"**Skills**: {', '.join(cand['skills'])}  \n"
                    f"**Work Experience**: \n- "
                    + "\n- ".join(cand["work_experience"])
                )
        else:
            st.warning("No suitable matches found.")

        st.markdown("---")
        st.markdown("### Workflow Summary")
        st.write(
            "JD Compared: ✅" if final_state.get("matched_resumes") else "JD Compared: ❌"
        )
        st.write(
            "Profiles Ranked: ✅" if final_state.get("ranked_resumes") else "Profiles Ranked: ❌"
        )
        st.write(
            "Email Sent: ✅" if final_state.get("emails") else "Email Sent: ❌"
        ) 