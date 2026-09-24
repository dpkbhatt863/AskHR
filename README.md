# 🏢 AskHR — AI Assistant for Employee Policies

A Retrieval-Augmented Generation (RAG) chatbot built with LangChain, ChromaDB, and Groq (Llama 3.3 70B) that answers employee questions based on company HR policy PDFs with page-level citations and an automated LLM-as-a-Judge evaluation framework.

---

## ⚡ Features

- **Document Q&A:** Answers employee questions strictly based on local HR policy PDFs.
- **Source Citations:** Every response includes the source filename and page number.
- **Persistent Chat History:** Q&A history stored locally using SQLite.
- **LLM-as-a-Judge Evaluation:** Built-in benchmarking suite measuring Context Relevance, Faithfulness (Groundedness), Correctness, and Latency.
- **No Hallucinations:** Refuses to answer out-of-scope questions if context is missing.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Frontend** | Streamlit |
| **LLM** | Groq (`openai/gpt-oss-120b`) |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2` - Local) |
| **Vector DB** | ChromaDB (Persistent) |
| **Metadata DB** | SQLite |
| **Orchestration** | LangChain |

---

## 📁 Project Structure

```
AskHR/
├── app.py                      # Streamlit frontend (Chat & Evaluation UI)
├── core/
│   ├── database.py             # SQLite helper for chat history
│   ├── document_processor.py   # PDF loader, chunking & ChromaDB vector store
│   ├── rag_chain.py            # Retrieval + Groq LLM chain
│   └── evaluator.py            # LLM-as-a-Judge evaluation logic
├── data/
│   ├── chroma_db/               # Vector database storage (auto-created)
│   ├── hr_app.db                 # SQLite database (auto-created)
│   └── *.pdf                    # HR Policy PDF files
├── eval/
│   └── test_dataset.json        # Test questions & expected answers
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install Dependencies

```bash
git clone <your-repository-url>
cd AskHR
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Add your free Groq API key (get one at [console.groq.com](https://console.groq.com/)):

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 3. Add Policy PDFs & Run

1. Place your company policy `.pdf` files inside the `data/` folder.
2. Start the application:

```bash
streamlit run app.py
```

3. On first launch, click **Index PDFs from Data Folder** to build the vector store.

---

## 📊 Evaluation Benchmark

The app includes an LLM-as-a-Judge framework (`core/evaluator.py`) using Groq to score outputs against ground-truth reference answers in `eval/test_dataset.json`:

- **Context Relevance:** Evaluates vector search chunk quality.
- **Faithfulness:** Verifies responses are strictly grounded in context without hallucination.
- **Answer Correctness:** Measures factual alignment against target reference answers.
