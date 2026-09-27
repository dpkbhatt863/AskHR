import json, os, time
from langchain_groq import ChatGroq
from core.rag_chain import ask_question

eval_llm = ChatGroq(model="qwen/qwen3.8-27b", api_key=os.getenv("GROQ_API_KEY"), temperature=0)


def judge_rag_turn(question, context, answer, reference, max_retries=2):
    """Uses LLM-as-a-Judge to evaluate Context Relevance, Faithfulness, and Correctness."""

    if not context:
        is_refusal = "could not find" in answer.lower()
        return {"context_relevance": 1.0 if is_refusal else 0.0,
                "faithfulness": 1.0 if is_refusal else 0.0,
                "correctness": 1.0 if is_refusal else 0.0,
                "reasoning": "No context retrieved; scored purely on whether the system refused."}

    prompt = f"""You are an expert RAG evaluator. Grade the following RAG system outputs on a scale from 0.0 to 1.0.

Question: {question}
Reference Answer: {reference}
Retrieved Context: {context}
Generated Answer: {answer}

Score these three dimensions:

1. Context Relevance (0.0-1.0): Does the retrieved context contain the information needed
   to answer the question? Score partially if only some of the required facts are present
   (e.g. 0.5 if the context covers half of a multi-part question).

2. Faithfulness (0.0-1.0): Is the generated answer strictly based ONLY on the retrieved
   context, without inventing facts not present there? A correct refusal ("I could not
   find an answer...") when the context is genuinely insufficient is fully faithful (1.0).

3. Correctness (0.0-1.0): Does the generated answer capture the KEY FACTS in the reference
   answer? Use PARTIAL CREDIT rather than an all-or-nothing judgment:
   - 1.0: All key facts from the reference are present and accurate.
   - 0.6-0.9: The core/headline fact is correct but one or more secondary details,
     conditions, exceptions, or sub-parts from the reference are missing or incomplete.
   - 0.1-0.5: Only a minor or tangential part of the reference is captured, OR the
     question has multiple distinct parts and only one part was meaningfully answered.
   - 0.0: The answer is wrong, contradicts the reference, OR the system refused to
     answer ("I could not find...") even though the reference shows an answer exists
     and was retrievable in principle.
   Do not penalize wording differences - judge factual and numerical equivalence only
   (e.g. "6 months" and "six (6) months" are equivalent).

Respond strictly in valid JSON format with keys "context_relevance", "faithfulness",
"correctness", and "reasoning" (a one-sentence explanation of the correctness score,
specifically naming any fact that was missing or wrong). Do not include Markdown code
blocks or any other text.
JSON format: {{"context_relevance": 0.0, "faithfulness": 0.0, "correctness": 0.0, "reasoning": "..."}}"""

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = eval_llm.invoke(prompt).content.strip()
            if response.startswith("```"):
                response = response.replace("```json", "").replace("```", "").strip()
            scores = json.loads(response)
            return {
                "context_relevance": float(scores.get("context_relevance", 0.0)),
                "faithfulness": float(scores.get("faithfulness", 0.0)),
                "correctness": float(scores.get("correctness", 0.0)),
                "reasoning": scores.get("reasoning", ""),
            }
        except Exception as e:
            last_error = e
            time.sleep(0.5)

    # Only reached if every attempt failed to parse - flag it instead of masking it
    # as a real 0.5 score, so failed evaluations are visible and distinguishable
    # from genuinely mediocre answers.
    return {
        "context_relevance": None,
        "faithfulness": None,
        "correctness": None,
        "reasoning": f"JUDGE_PARSE_FAILURE after {max_retries + 1} attempts: {last_error}",
    }


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
    parse_failures = 0

    for tc in test_questions:
        question = tc["question"]
        ref_answer = tc["expected_answer"]  # Mapped from your JSON key

        # 1. Run RAG Pipeline
        rag = ask_question(question)

        # 2. Extract Context Text
        context_str = "\n".join(s["text"] for s in rag["sources"])

        # 3. Grade using LLM-as-a-Judge
        scores = judge_rag_turn(question, context_str, rag["answer"], ref_answer)

        if scores["context_relevance"] is None:
            parse_failures += 1
        else:
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
            "difficulty": tc.get("difficulty", "unknown"),
        })

    return {
        "summary": {
            "total": len(test_questions),
            "judge_parse_failures": parse_failures,
            "avg_context_relevance": round(sum(rel_scores) / len(rel_scores), 2) if rel_scores else 0,
            "avg_faithfulness": round(sum(faith_scores) / len(faith_scores), 2) if faith_scores else 0,
            "avg_correctness": round(sum(corr_scores) / len(corr_scores), 2) if corr_scores else 0,
            "avg_latency": round(sum(latencies) / len(latencies), 3) if latencies else 0,
        },
        "details": results,
    }