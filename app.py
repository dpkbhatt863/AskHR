import streamlit as st
from dotenv import load_dotenv

from core.database import init_db, add_chat_entry, get_chat_history, clear_chat_history
from core.document_processor import load_and_index_policies, collection_count
from core.rag_chain import ask_question
from core.evaluator import run_full_evaluation

# Initialize environment and DB
load_dotenv()
init_db()

st.set_page_config(page_title="AskHR", layout="centered")

# Initialize session state for persisted evaluation
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

# ── Sidebar Navigation ────────────────────────────────────────────────────────
st.sidebar.title("AskHR")
page = st.sidebar.radio("Go to", ["💬 Chat", "📊 Evaluator"])

# ── Page 1: Chat ──────────────────────────────────────────────────────────────
if page == "💬 Chat":
    st.title("💬 AskHR")

    if collection_count() == 0:
        st.warning("Policies have not been indexed yet.")
        if st.button("Index PDFs from Data Folder"):
            with st.spinner("Indexing PDFs... this only happens once."):
                load_and_index_policies()
            st.success("Indexing complete!")
            st.rerun()
            
    else:
        if st.button("🗑️ Clear History"):
            clear_chat_history()
            st.rerun()

        # Display history
        for entry in get_chat_history():
            with st.chat_message("user"):
                st.markdown(entry["question"])
            with st.chat_message("assistant"):
                st.markdown(entry["answer"])
                if entry["sources"]:
                    with st.expander("📚 Sources", expanded=False):
                        for s in entry["sources"]:
                            st.write(f"- **{s['source']}** (Page {s['page']})")

        # Chat input
        if question := st.chat_input("Ask a question about HR policies..."):
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching..."):
                    res = ask_question(question)
                
                st.markdown(res["answer"])
                
                if res["sources"]:
                    with st.expander("📚 Sources", expanded=False):
                        for s in res["sources"]:
                            st.write(f"- **{s['source']}** (Page {s['page']})")
                            st.caption(f'"{s["text"]}"')

            add_chat_entry(question, res["answer"], res["sources"])

# ── Page 2: Evaluator ─────────────────────────────────────────────────────────
elif page == "📊 Evaluator":
    st.title("📊 Evaluator")
    
    if collection_count() == 0:
        st.error("Please go to the Chat tab and index the policies first.")
    else:
        # Button to run or refresh evaluation
        btn_label = "🚀 Run LLM Evaluation" if st.session_state.eval_results is None else "🔄 Re-run Evaluation"
        
        if st.button(btn_label):
            with st.spinner("Running evaluation suite..."):
                st.session_state.eval_results = run_full_evaluation()
            st.rerun()

        # Display persisted results immediately if they exist in session_state
        if st.session_state.eval_results:
            res = st.session_state.eval_results
            if "error" in res:
                st.error(res["error"])
            else:
                s = res["summary"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Context Relevance", f"{s['avg_context_relevance']:.0%}")
                c2.metric("Faithfulness", f"{s['avg_faithfulness']:.0%}")
                c3.metric("Correctness", f"{s['avg_correctness']:.0%}")
                c4.metric("Avg Latency", f"{s['avg_latency']}s")
                
                st.divider()
                
                for r in res["details"]:
                    sc = r["scores"]
                    with st.expander(f"Q: {r['question']}"):
                        st.write(f"**Generated:** {r['answer']}")
                        st.write(f"**Reference:** {r['reference']}")
                        st.caption(
                            f"Relevance: {sc['context_relevance']} | "
                            f"Faithfulness: {sc['faithfulness']} | "
                            f"Correctness: {sc['correctness']}"
                        )
        else:
            st.info("Click the button above to run the evaluation suite.")