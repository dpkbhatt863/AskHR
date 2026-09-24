import json, os, time
from langchain_groq import ChatGroq
from core.rag_chain import ask_question

eval_llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"), temperature=0)

def judge_rag_turn(question, context, answer, reference):
    """Uses LLM-as-a-Judge to evaluate Context Relevance, Faithfulness, and Correctness."""
    
    if not context:
        is_refusal = "could not find" in answer.lower()
        return {"context_relevance": 1.0 if is_refusal else 0.0,
                "faithfulness": 1.0 if is_refusal else 0.0,
                "correctness": 1.0 if is_refusal else 0.0}

    prompt = f"""You are an expert RAG evaluator. Grade the following RAG system outputs on a scale from 0.0 to 1.0.

Question: {question}
Reference Answer: {reference}
Retrieved Context: {context}
Generated Answer: {answer}

Provide ratings for:
1. Context Relevance: Does the retrieved context contain the information needed to answer the question? (0.0 to 1.0)
2. Faithfulness: Is the generated answer strictly based ONLY on the retrieved context without hallucination? (0.0 to 1.0)
3. Correctness: Does the generated answer accurately match the key facts of the reference answer? (0.0 to 1.0)

Respond strictly in valid JSON format with keys "context_relevance", "faithfulness", "correctness". Do not include Markdown code blocks or any other text.
JSON format: {{"context_relevance": 0.0, "faithfulness": 0.0, "correctness": 0.0}}"""

    try:
        response = eval_llm.invoke(prompt).content.strip()
        if response.startswith("```json"):
            response = response.replace("```json", "").replace("```", "").strip()
        scores = json.loads(response)
        return {
            "context_relevance": float(scores.get("context_relevance", 0.0)),
            "faithfulness": float(scores.get("faithfulness", 0.0)),
            "correctness": float(scores.get("correctness", 0.0))
        }
    except Exception:
        return {"context_relevance": 0.5, "faithfulness": 0.5, "correctness": 0.5}


def run_full_evaluation():
    path = "eval/test_dataset.json"
    if not os.path.exists(path):
        return {"error": "Test dataset not found at eval/test_dataset.json"}

    with open(path) as f:
        full_dataset = json.load(f)
        
    # Extract the array of questions from your specific Nexora JSON structure
    test_questions = full_dataset.get("questions", [])
    
    if not test_questions:
        return {"error": "No questions found in the dataset. Check JSON format."}

    results, rel_scores, faith_scores, corr_scores, latencies = [], [], [], [], []

    for tc in test_questions:
        question = tc["question"]
        ref_answer = tc["expected_answer"] # Mapped from your JSON key
        
        # 1. Run RAG Pipeline
        rag = ask_question(question)
        
        # 2. Extract Context Text
        context_str = "\n".join(s["text"] for s in rag["sources"])
        
        # 3. Grade using LLM-as-a-Judge
        scores = judge_rag_turn(question, context_str, rag["answer"], ref_answer)
        
        rel_scores.append(scores["context_relevance"])
        faith_scores.append(scores["faithfulness"])
        corr_scores.append(scores["correctness"])
        latencies.append(rag["latency"])

        results.append({
            "question": question,
            "answer": rag["answer"],
            "reference": ref_answer,
            "scores": scores,
            "latency": rag["latency"],
            "difficulty": tc.get("difficulty", "unknown") # Keep track of difficulty
        })

    return {
        "summary": {
            "total": len(test_questions),
            "avg_context_relevance": round(sum(rel_scores) / len(rel_scores), 2) if rel_scores else 0,
            "avg_faithfulness": round(sum(faith_scores) / len(faith_scores), 2) if faith_scores else 0,
            "avg_correctness": round(sum(corr_scores) / len(corr_scores), 2) if corr_scores else 0,
            "avg_latency": round(sum(latencies) / len(latencies), 3) if latencies else 0,
        },
        "details": results,
    }