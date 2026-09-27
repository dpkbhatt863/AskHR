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
2. Be COMPLETE, not just correct. Include every relevant condition, exception, deadline,
   monetary figure, and sub-clause found in the context that relates to the question -
   not only the single headline fact. If the reference a user would expect includes
   related details (e.g. an extension period, who is covered, a time limit), surface
   those too, even if the question did not explicitly ask for them.
3. If the question has multiple distinct parts (e.g. joined by "and", "separately", or
   multiple question marks), answer EACH part explicitly and clearly label them, using
   whatever relevant context is available for each part.
4. If the context answers only SOME parts of the question, answer the parts you can
   support from the context, and explicitly say which specific part you could not find
   information for. Only use the full refusal message in Rule 5 if NONE of the question
   can be answered from the context.
5. If the context does not contain the answer at all, reply exactly: "I could not find
   an answer to this question in the uploaded HR policy documents."
6. Keep answers clear, professional, and factual. Do not assume or extrapolate beyond
   the context.

Context:
{context}"""),
    ("human", "{question}"),
])

def ask_question(question, top_k=6):
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