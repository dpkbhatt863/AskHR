import os
import streamlit as st
from dotenv import load_dotenv

from core.database import init_db, add_chat_entry, get_chat_history, clear_chat_history
from core.document_processor import load_and_index_policies, collection_count
from core.rag_chain import ask_question
from core.evaluator import run_full_evaluation

load_dotenv()
init_db()

st.set_page_config(page_title="AskHR", layout="centered")

if collection_count() == 0:
    with st.spinner("Initializing HR policy knowledge base..."):
        load_and_index_policies()

SAMPLES = [
    "What are the standard working hours?",
    "How many casual leave days do I get?",
    "What is the probation period?",
    "What is the health insurance coverage?",
    "What is the notice period for resignation?",
    "Can I accept a gift from a vendor?",
]

def show_evaluator():
    val = os.getenv("SHOW_EVALUATOR", "")
    try:
        val = val or st.secrets.get("SHOW_EVALUATOR", "")
    except Exception:
        pass
    return str(val).lower() in ("1", "true", "yes")

if "pending" not in st.session_state:
    st.session_state.pending = None
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None
if "page" not in st.session_state:
    st.session_state.page = "chat"

#sidebar section
with st.sidebar:
    st.title("AskHR")
    st.caption("AI assistant for company HR policies")

    st.markdown(
        "Ask about leave, working hours, benefits, "
        "notice period, code of conduct, and more. "
        "Answers are grounded in official policy PDFs with source citations."
    )

    st.divider()
    st.markdown("**Quick asks**")
    for i, q in enumerate(SAMPLES):
        if st.button(q, key=f"side_q_{i}", use_container_width=True):
            st.session_state.pending = q
            st.session_state.page = "chat"
            st.rerun()

    st.divider()
    st.caption(f"🟢 {collection_count()} policy chunks ready")
    st.caption("For case-specific decisions, contact HR.")

    if show_evaluator():
        st.divider()
        if st.button("Open Evaluator", use_container_width=True):
            st.session_state.page = "eval"
            st.rerun()
        if st.session_state.page == "eval":
            if st.button("Back to Chat", use_container_width=True):
                st.session_state.page = "chat"
                st.rerun()


#chat section
if st.session_state.page == "chat" or not show_evaluator():
    st.title("AskHR")
    st.caption("Your AI assistant for leave, benefits, attendance, and workplace policies.")

    if st.button("Clear chat"):
        clear_chat_history()
        st.rerun()

    # Render previous conversation history
    for entry in get_chat_history():
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])
            if entry["sources"]:
                with st.expander("Sources"):
                    for s in entry["sources"]:
                        st.write(f"**{s['source']}**, page {s['page']}")
                        st.caption(s["text"][:300])

    # Input logic
    question = st.session_state.pending or st.chat_input("Ask about an HR policy...")
    st.session_state.pending = None

    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Checking policy documents..."):
                res = ask_question(question)
            st.markdown(res["answer"])
            if res["sources"]:
                with st.expander("Sources"):
                    for s in res["sources"]:
                        st.write(f"**{s['source']}**, page {s['page']}")
                        st.caption(s["text"][:300])
        add_chat_entry(question, res["answer"], res["sources"])
        st.rerun()


#evaluator section (only for local run)
elif st.session_state.page == "eval" and show_evaluator():
    st.title("Evaluator")
    st.caption("Local demo only. Runs the full test set and can take a minute.")

    label = "Run evaluation" if st.session_state.eval_results is None else "Re-run evaluation"
    if st.button(label):
        with st.spinner("Running test dataset evaluation..."):
            st.session_state.eval_results = run_full_evaluation()
        st.rerun()

    res = st.session_state.eval_results
    if not res:
        st.info("Click the button above to run the evaluation.")
    elif "error" in res:
        st.error(res["error"])
    else:
        s = res["summary"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Context relevance", f"{s['avg_context_relevance']:.0%}")
        c2.metric("Faithfulness", f"{s['avg_faithfulness']:.0%}")
        c3.metric("Correctness", f"{s['avg_correctness']:.0%}")
        c4.metric("Avg latency", f"{s['avg_latency']}s")
        st.divider()
        for r in res["details"]:
            sc = r["scores"]
            with st.expander(r["question"]):
                st.write(r["answer"])
                st.caption(
                    f"Relevance {sc['context_relevance']} | "
                    f"Faithfulness {sc['faithfulness']} | "
                    f"Correctness {sc['correctness']}"
                )