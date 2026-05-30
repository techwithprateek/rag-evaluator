# config.py — Central provider configuration
# Reads the PROVIDER env variable and returns the right LLM + embeddings.
# This lets you swap between OpenAI (paid, fast) and HuggingFace (free, slower)
# without touching any other file.

import os
from dotenv import load_dotenv

# Load variables from the .env file into the environment
load_dotenv()

# Read the provider choice; default to "openai" if not set
PROVIDER = os.getenv("PROVIDER", "openai").lower()


def get_llm():
    """
    Return a LangChain-compatible LLM based on the selected provider.
    temperature=0 means deterministic outputs — important for evaluation
    consistency. You want the same question to produce the same answer.
    """
    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is required when PROVIDER=openai")
        # gpt-4o-mini is fast and cheap — ideal for this kind of pipeline
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    elif PROVIDER == "huggingface":
        from langchain_huggingface import HuggingFaceEndpoint
        if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
            raise EnvironmentError("HUGGINGFACEHUB_API_TOKEN is required when PROVIDER=huggingface")
        # HuggingFaceEndpoint calls the HF Inference API — no local GPU needed
        # Mistral-7B is a strong open-source instruction-following model
        return HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            temperature=0.0,
            max_new_tokens=512,
        )

    else:
        raise ValueError(f"Unknown provider '{PROVIDER}'. Use 'openai' or 'huggingface'.")


def get_embeddings():
    """
    Return a LangChain-compatible embeddings model based on the selected provider.
    Embeddings convert text → dense vectors so we can do similarity search in ChromaDB.
    The same embedding model must be used during ingestion AND retrieval.
    """
    if PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is required when PROVIDER=openai")
        # text-embedding-3-small: great balance of quality and cost
        return OpenAIEmbeddings(model="text-embedding-3-small")

    elif PROVIDER == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        # all-MiniLM-L6-v2: runs locally, no API calls, fast and surprisingly good
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    else:
        raise ValueError(f"Unknown provider '{PROVIDER}'. Use 'openai' or 'huggingface'.")
