import os, time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from core.document_processor import retrieve_relevant_chunks

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"), temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are AskHR, an AI assistant for company employee policies.
Answer the user's question strictly using ONLY the provided HR policy context.

Rules:
1. Cite the source document and page number for every claim, e.g. (Leave_Policy.pdf, Page 2).
2. If the context does not contain the answer, reply exactly: "I could not find an answer to this question in the uploaded HR policy documents."
3. Keep answers clear, professional, and factual. Do not assume or extrapolate.

Context:
{context}"""),
    ("human", "{question}"),
])

def ask_question(question, top_k=5):
    start = time.time()
    chunks = retrieve_relevant_chunks(question, top_k)

    if not chunks:
        return {
            "answer": "No HR policy documents uploaded yet. Please upload policy documents first.",
            "sources": [],
            "latency": round(time.time() - start, 3),
        }

    context = "\n\n---\n\n".join(f"[{c['source']}, Page {c['page']}]\n{c['text']}" for c in chunks)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    return {
        "answer": response.content,
        "sources": chunks,
        "latency": round(time.time() - start, 3),
    }