# evaluate.py — Step 3: Score the RAG pipeline with RAGAS
#
# RAGAS is an "LLM-as-a-judge" evaluation framework. It uses an LLM to score
# how well your RAG pipeline performed — no human labeling needed beyond ground truths.
#
# It reads the outputs saved by rag_pipeline.py and scores each sample on 4 metrics:
#   - Faithfulness:      Is the answer grounded in the retrieved context?
#   - Answer Relevancy:  Does the answer actually address the question?
#   - Context Precision: Were the retrieved chunks relevant (low noise)?
#   - Context Recall:    Did retrieval find everything needed to answer?

import json
import os
from datasets import Dataset
from ragas import evaluate

# Import the 4 metric singleton instances from RAGAS
# These are pre-configured metric objects that know how to score each dimension
from ragas.metrics._faithfulness import faithfulness
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall

# RAGAS needs its own LLM/embeddings wrappers — it doesn't use LangChain objects directly.
# LangchainLLMWrapper adapts a LangChain LLM so RAGAS can use it as the judge.
# LangchainEmbeddingsWrapper does the same for embeddings (needed by Answer Relevancy).
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from config import get_llm, get_embeddings, PROVIDER

OUTPUTS_FILE = "outputs/rag_outputs.json"
RESULTS_FILE = "results/ragas_results.csv"

# All 4 metrics to run — passed as a list to evaluate()
METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


def run_evaluation():
    if not os.path.exists(OUTPUTS_FILE):
        raise FileNotFoundError(f"Outputs not found at '{OUTPUTS_FILE}'. Run rag_pipeline.py first.")

    os.makedirs("results", exist_ok=True)

    # Load the JSON saved by rag_pipeline.py
    # Each entry has: question, answer, contexts (list), ground_truth
    with open(OUTPUTS_FILE) as f:
        outputs = json.load(f)
    print(f"Loaded {len(outputs)} samples from '{OUTPUTS_FILE}'")

    # Wrap our LangChain LLM and embeddings so RAGAS can use them as the evaluator.
    # The same LLM that generated answers can also act as the judge — this is fine
    # for learning, but in production you'd use a separate, stronger judge model.
    print(f"Configuring RAGAS with provider: {PROVIDER}")
    ragas_llm = LangchainLLMWrapper(get_llm())
    ragas_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    # RAGAS expects a Hugging Face Dataset object.
    # The column names must match exactly: question, answer, contexts, ground_truth.
    dataset = Dataset.from_list(outputs)

    print("Running RAGAS evaluation (this may take a few minutes)...\n")

    # evaluate() sends each sample to the LLM judge for scoring.
    # It runs all 4 metrics in parallel across all samples — hence the wait.
    # llm = the judge model, embeddings = used by answer_relevancy to measure
    # semantic similarity between the question and generated answer.
    results = evaluate(
        dataset=dataset,
        metrics=METRICS,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    # Convert to a pandas DataFrame for easy inspection and saving
    df = results.to_pandas()
    df.to_csv(RESULTS_FILE, index=False)

    # Print the mean score for each metric across all 10 samples
    print("=" * 50)
    print("        RAGAS Evaluation Summary")
    print("=" * 50)
    print(f"  Faithfulness:       {df['faithfulness'].mean():.4f}")
    print(f"  Answer Relevancy:   {df['answer_relevancy'].mean():.4f}")
    print(f"  Context Precision:  {df['context_precision'].mean():.4f}")
    print(f"  Context Recall:     {df['context_recall'].mean():.4f}")
    print("=" * 50)
    print(f"\n✓ Full results saved to '{RESULTS_FILE}'")


if __name__ == "__main__":
    run_evaluation()
