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

SAMPLES = [
    "What are the standard working hours?",
    "How many casual leave days do I get per year?",
    "What is the probation period for a new employee?",
    "What is the health insurance coverage?",
]

def show_evaluator():
    # Set SHOW_EVALUATOR=true in local .env only. Do not set it on Streamlit Cloud.
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

pages = ["Chat", "Evaluator"] if show_evaluator() else ["Chat"]
page = st.sidebar.radio("Go to", pages)
st.sidebar.caption("Answers come only from company HR policy documents.")

if page == "Chat":
    st.title("AskHR")
    st.caption("Ask about leave, working hours, benefits, or notice period.")

    if collection_count() == 0:
        st.warning("Policies are not indexed yet.")
        if st.button("Index PDFs"):
            with st.spinner("Indexing policies..."):
                load_and_index_policies()
            st.rerun()
    else:
        c1, c2 = st.columns([1, 4])
        if c1.button("Clear chat"):
            clear_chat_history()
            st.rerun()

        st.write("Try a sample question")
        cols = st.columns(2)
        for i, q in enumerate(SAMPLES):
            if cols[i % 2].button(q, key=f"sample_{i}"):
                st.session_state.pending = q
                st.rerun()

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

        question = st.session_state.pending or st.chat_input("Ask about an HR policy...")
        st.session_state.pending = None

        if question:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Checking the policy documents..."):
                    res = ask_question(question)
                st.markdown(res["answer"])
                if res["sources"]:
                    with st.expander("Sources"):
                        for s in res["sources"]:
                            st.write(f"**{s['source']}**, page {s['page']}")
                            st.caption(s["text"][:300])
            add_chat_entry(question, res["answer"], res["sources"])
            st.rerun()

elif page == "Evaluator":
    st.title("Evaluator")
    st.caption("Local demo only. Runs the full test set and can take a minute.")

    if collection_count() == 0:
        st.error("Index the policies from the Chat page first.")
    else:
        label = "Run evaluation" if st.session_state.eval_results is None else "Re-run evaluation"
        if st.button(label):
            with st.spinner("Running the test questions. This can take a minute."):
                st.session_state.eval_results = run_full_evaluation()
            st.rerun()

        res = st.session_state.eval_results
        if not res:
            st.info("Run the evaluation when you want fresh scores.")
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