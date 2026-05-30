# ingest.py — Step 1: Load → Chunk → Embed → Store
#
# This script builds the knowledge base that the RAG pipeline will query.
# Run this once before rag_pipeline.py. Re-running it will wipe and rebuild
# the vector store from scratch to avoid duplicate embeddings.

import os
import time
import wikipedia as wiki_client
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from config import get_embeddings

# The 5 Wikipedia topics that form our knowledge base
TOPICS = [
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    "Natural language processing",
    "Large language model",
]

# Local folder where ChromaDB persists the vector index
CHROMA_PATH = "chroma_db"


def fetch_wikipedia_page(topic: str, retries: int = 3) -> Document | None:
    """
    Fetch a Wikipedia article and wrap it in a LangChain Document.
    
    We set a user-agent because Wikipedia's API rejects requests without one.
    auto_suggest=False ensures we get exactly the page we asked for, not a
    "did you mean X?" redirect that could return unrelated content.
    Retries handle transient network errors or Wikipedia rate-limiting.
    """
    wiki_client.set_user_agent("rag-evaluator/1.0 (educational project)")
    for attempt in range(retries):
        try:
            page = wiki_client.page(topic, auto_suggest=False)
            # Document is LangChain's standard container: text + metadata
            return Document(
                page_content=page.content,
                metadata={"source": page.url, "title": page.title},
            )
        except Exception as e:
            print(f"  ⚠ Attempt {attempt+1} failed for '{topic}': {e}")
            time.sleep(2)  # wait before retrying to avoid hammering the API
    return None


def ingest():
    print("Loading Wikipedia articles...")
    docs = []
    for topic in TOPICS:
        doc = fetch_wikipedia_page(topic)
        if doc:
            docs.append(doc)
            print(f"  ✓ Loaded: {topic}")
        else:
            print(f"  ✗ Skipped: {topic} (all retries failed)")
        time.sleep(1)  # polite delay — Wikipedia asks for respectful crawling

    print(f"\nTotal documents loaded: {len(docs)}")

    # --- Chunking ---
    # LLMs have a limited context window, so we can't feed entire Wikipedia articles.
    # We split each article into smaller overlapping chunks.
    # chunk_size=500: each chunk is ~500 characters
    # chunk_overlap=50: chunks share 50 characters at edges, so context isn't cut off mid-sentence
    # "Recursive" means it tries to split on paragraphs first, then sentences, then words.
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"Total chunks after splitting: {len(chunks)}")

    print("\nEmbedding and storing in ChromaDB...")
    embeddings = get_embeddings()

    # Wipe the existing DB on re-run to avoid duplicate vectors
    if os.path.exists(CHROMA_PATH):
        import shutil
        shutil.rmtree(CHROMA_PATH)

    # Chroma.from_documents:
    # 1. Calls the embedding model on every chunk to get a vector
    # 2. Stores both the text and its vector in the local ChromaDB
    # persist_directory saves the index to disk so we don't re-embed on every run
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )

    print(f"✓ ChromaDB created at '{CHROMA_PATH}' with {len(chunks)} vectors")


if __name__ == "__main__":
    ingest()
