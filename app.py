import os
import tempfile
import asyncio

import streamlit as st

# MongoDB helpers
from db_helper import fetch_all_results, get_pdf_bytes, insert_job_result

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
    
    # We are automating the clearing of the vector store, so this button is no longer needed.
    # if st.button("🗑️ Clear Vector Store", help="Remove all stored resumes"):
    #     clear_vector_store()
    #     st.success("Vector store cleared!")
    #     st.rerun()

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
# Main content with tabs
# ------------------------------------------------------------
tab1, tab2 = st.tabs(["🚀 New Match Request", "📋 View History"])

# ------------------------------------------------------------
# Tab 1: New Match Request
# ------------------------------------------------------------
with tab1:
    st.header("Match Job Description with Resumes")
    
    jd_text = st.text_area("Job Description", height=180, help="Enter the job description you want to match against resumes")

    uploaded_files = st.file_uploader(
        "Upload Resume PDFs", type=["pdf"], accept_multiple_files=True,
        help="Select one or more PDF resume files"
    )

    run_button = st.button("🚀 Run Comparison", disabled=not (jd_text and uploaded_files), type="primary")

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
                # 1️⃣ Clear the vector store before starting
                status_placeholder.info("Clearing vector store for a fresh run...")
                clear_vector_store()
                
                # 2️⃣ Extract & store resumes
                status_placeholder.info("Extracting resumes & building vector index …")
                state.update(await extract_and_store_resumes(state))
                progress_bar.progress(0.33)

                # 3️⃣ Compare JD → vector search
                status_placeholder.info("Comparing JD with resumes …")
                state.update(await comparison_agent(state))

                # 4️⃣ Rank via LLM
                status_placeholder.info("Ranking profiles …")
                state.update(await ranking_agent(state))
                # 4b️⃣ Persist to MongoDB
                try:
                    insert_job_result(
                        jd=state["jd"],
                        ranked_resumes=state.get("ranked_resumes", []),
                        resume_file_path_mapping=state.get("resume_file_path_mapping", {}),
                    )
                    print("[MongoDB] Results stored successfully (Streamlit workflow)")
                except Exception as e:
                    print(f"[MongoDB] Failed to store results from Streamlit workflow: {e}")
                progress_bar.progress(0.66)

                # 5️⃣ Send email
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
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(
                                f"**{i}. {cand['name']}**  \n"
                                f":e-mail: {cand['email']}  \n"
                                f"**Match Score**: {cand.get('matched_score', 'N/A')}  \n"
                                f"**Skills**: {', '.join(cand['skills'])}  \n"
                                f"**Work Experience**: \n- "
                                + "\n- ".join(cand["work_experience"])
                            )
                            if cand.get('rank_reason'):
                                st.markdown(f"**Ranking Reason**: {cand['rank_reason']}")
                        st.divider()
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

# ------------------------------------------------------------
# Tab 2: View History
# ------------------------------------------------------------
with tab2:
    st.header("Previous Matching Results")
    
    past_results = fetch_all_results()
    
    if past_results:
        st.write(f"Found **{len(past_results)}** previous matching sessions")
        st.markdown("---")
        
        for idx, run in enumerate(past_results):
            created_at = run.get("created_at", "Unknown date")
            jd_text = run.get("jd", "")
            ranked_resumes = run.get("ranked_resumes", [])
            
            # Create a card-like container for each run
            with st.container():
                st.markdown(f"### 📄 Session #{idx + 1}")
                st.caption(f"Created: {created_at}")
                
                # Job Description section
                with st.expander("📋 View Job Description", expanded=False):
                    st.text_area("Job Description", value=jd_text, height=150, disabled=True, key=f"jd_{idx}")
                
                # Resume Results section
                st.markdown("#### 🎯 Matched Resumes")
                
                if ranked_resumes:
                    for rank_idx, cand in enumerate(ranked_resumes, 1):
                        with st.container():
                            col1, col2 = st.columns([4, 1])
                            
                            with col1:
                                st.markdown(f"**{rank_idx}. {cand.get('name', 'Unknown')}**")
                                st.write(f"📧 **Email**: {cand.get('email', 'N/A')}")
                                
                                # Skills
                                skills = cand.get('skills', [])
                                if skills:
                                    st.write(f"🔧 **Skills**: {', '.join(skills)}")

                                # Match Score
                                match_score = cand.get('matched_score')
                                if match_score is not None:
                                    # Score already comes in percentage format
                                    st.write(f"🎯 **Match Score**: {match_score}")
                                
                                # Work Experience
                                work_exp = cand.get('work_experience', [])
                                if work_exp:
                                    st.write("💼 **Work Experience**:")
                                    for exp in work_exp[:2]:  # Show first 2 experiences
                                        st.write(f"   • {exp}")
                                    if len(work_exp) > 2:
                                        st.write(f"   • ... and {len(work_exp) - 2} more")
                                
                                # Ranking reason
                                rank_reason = cand.get('rank_reason')
                                if rank_reason:
                                    st.write(f"📈 **Ranking Reason**: {rank_reason}")
                            
                            with col2:
                                # Download button for PDF
                                pdf_id = cand.get("pdf_file_id")
                                if pdf_id:
                                    pdf_bytes = get_pdf_bytes(pdf_id)
                                    if pdf_bytes:
                                        st.download_button(
                                            label="📥 Download PDF",
                                            data=pdf_bytes,
                                            file_name=f"{cand.get('name', 'resume')}.pdf",
                                            mime="application/pdf",
                                            key=f"download_{idx}_{rank_idx}",
                                            help=f"Download {cand.get('name', '')}'s resume"
                                        )
                                    else:
                                        st.write("❌ PDF not available")
                                else:
                                    st.write("📄 No PDF stored")
                            
                            st.divider()
                    
                else:
                    st.warning("No resumes found for this session")
                
                st.markdown("---")
                
    else:
        st.info("📭 No previous matching sessions found. Start by creating a new match request in the first tab!")
        st.markdown("---")
        st.markdown("💡 **Tip**: Once you run a job description matching session, the results will appear here for future reference.") 