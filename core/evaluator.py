import json, os
from core.rag_chain import ask_question

def run_full_evaluation():
    path = "eval/test_dataset.json"
    if not os.path.exists(path):
        return {"error": "Test dataset not found at eval/test_dataset.json"}

    with open(path) as f:
        test_data = json.load(f)

    results, coverages, groundedness_scores, latencies = [], [], [], []

    for tc in test_data:
        question = tc["question"]
        keywords = tc.get("expected_keywords", [])
        rag = ask_question(question)
        answer_lower = rag["answer"].lower()

        # 1. Retrieval keyword coverage
        all_retrieved_text = " ".join(s["text"].lower() for s in rag["sources"])
        if keywords:
            coverage = sum(1 for k in keywords if k.lower() in all_retrieved_text) / len(keywords)
            coverages.append(coverage)
        else:
            coverage = None  # Out-of-scope question

        # 2. Groundedness check
        has_citation = any(s["source"].lower().replace(".pdf", "") in answer_lower for s in rag["sources"])
        refused = "could not find" in answer_lower
        grounded = 1.0 if (has_citation or (not keywords and refused)) else 0.0
        groundedness_scores.append(grounded)

        # 3. Latency
        latencies.append(rag["latency"])

        results.append({
            "question": question,
            "answer": rag["answer"],
            "coverage": coverage,
            "groundedness": grounded,
            "latency": rag["latency"],
        })

    return {
        "summary": {
            "total": len(test_data),
            "avg_coverage": round(sum(coverages) / len(coverages), 3) if coverages else 0,
            "avg_groundedness": round(sum(groundedness_scores) / len(groundedness_scores), 3),
            "avg_latency": round(sum(latencies) / len(latencies), 3),
        },
        "details": results,
    }