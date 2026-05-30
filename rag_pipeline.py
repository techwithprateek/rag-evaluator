# rag_pipeline.py — Step 2: Retrieve → Generate → Save
#
# This is the actual RAG pipeline. For each pre-generated question it:
#   1. Embeds the question and finds the 3 most similar chunks (Retrieve)
#   2. Passes those chunks + the question to the LLM (Generate)
#   3. Saves question, answer, retrieved contexts, and ground truth to JSON
#
# The saved JSON is what RAGAS will evaluate in evaluate.py.

import json
import os
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from config import get_llm, get_embeddings

CHROMA_PATH = "chroma_db"
QUESTIONS_FILE = "data/questions.json"
OUTPUTS_FILE = "outputs/rag_outputs.json"

# The prompt constrains the LLM to only use retrieved context.
# "If the answer is not in the context, say I don't know" prevents hallucination
# and keeps faithfulness scores honest — we don't want the model to rely on
# its own training data instead of what was retrieved.
PROMPT_TEMPLATE = """Use only the following context to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""


def format_docs(docs):
    """Join retrieved Document objects into a single context string for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def run_pipeline():
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(f"ChromaDB not found at '{CHROMA_PATH}'. Run ingest.py first.")

    # Load the pre-generated questions and their ground truth answers
    with open(QUESTIONS_FILE) as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions from '{QUESTIONS_FILE}'")

    # Load the existing ChromaDB index — no re-embedding needed
    embeddings = get_embeddings()
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # as_retriever wraps ChromaDB in LangChain's retriever interface.
    # k=3 means: fetch the 3 most semantically similar chunks for each query.
    # More chunks = more context but also more noise — 3 is a good starting point.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # --- LangChain LCEL Chain ---
    # LCEL (LangChain Expression Language) uses the | pipe operator to chain steps.
    #
    # retrieve_chain runs two things in parallel for a given question:
    #   - docs: fetches the top-3 relevant chunks from ChromaDB
    #   - question: passes the question string through unchanged (RunnablePassthrough)
    retrieve_chain = RunnableParallel(
        docs=retriever,
        question=RunnablePassthrough(),
    )

    # answer_chain takes the output of retrieve_chain and:
    #   1. Formats docs into a context string + keeps the question
    #   2. Fills the prompt template
    #   3. Sends to the LLM
    #   4. Parses the raw LLM response into a plain string (StrOutputParser)
    answer_chain = (
        {"context": lambda x: format_docs(x["docs"]), "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )

    results = []
    for i, item in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {item['question']}")

        # Step 1: retrieve relevant chunks for this question
        retrieved = retrieve_chain.invoke(item["question"])

        # Step 2: generate an answer using the retrieved chunks as context
        answer = answer_chain.invoke(retrieved)

        # Save all 4 fields — this is the exact format RAGAS expects:
        # question, answer, contexts (list of strings), ground_truth
        results.append({
            "question": item["question"],
            "answer": answer.strip(),
            "contexts": [doc.page_content for doc in retrieved["docs"]],
            "ground_truth": item["ground_truth"],
        })

    os.makedirs("outputs", exist_ok=True)
    with open(OUTPUTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ RAG outputs saved to '{OUTPUTS_FILE}'")


if __name__ == "__main__":
    run_pipeline()
