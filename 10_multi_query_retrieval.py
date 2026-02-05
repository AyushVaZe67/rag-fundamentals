from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

load_dotenv()

# ──────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────

persistent_directory = "db/chroma_db"

# Hugging Face Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Groq LLM
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# Chroma Vector DB
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# ──────────────────────────────────────────────────────────────────
# Pydantic model for structured output
# ──────────────────────────────────────────────────────────────────

class QueryVariations(BaseModel):
    queries: List[str]

# ──────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────

original_query = "How does Tesla make money?"
print(f"Original Query: {original_query}\n")

# Structured output with Groq
llm_with_tools = llm.with_structured_output(QueryVariations)

prompt = f"""
Generate 3 different variations of this query that would help retrieve relevant documents.

Original query: {original_query}

Return 3 alternative queries that rephrase or approach the same question from different angles.
"""

response = llm_with_tools.invoke(prompt)

query_variations = response.queries

print("Generated Query Variations:")
for i, variation in enumerate(query_variations, 1):
    print(f"{i}. {variation}")

print("\n" + "=" * 60)