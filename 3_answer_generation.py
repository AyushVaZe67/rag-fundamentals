from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Chroma DB directory
persistent_directory = "db/chroma_db"

# -------------------------------
# Embedding Model (HuggingFace)
# -------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------
# Load Vector Store
# -------------------------------
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# -------------------------------
# User Query
# -------------------------------
query = "How much did Microsoft pay to acquire GitHub?"

# -------------------------------
# Retriever
# -------------------------------
retriever = db.as_retriever(
    search_kwargs={"k": 5}
)

# Optional threshold-based retriever
# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3
#     }
# )

relevant_docs = retriever.invoke(query)

# -------------------------------
# Display Retrieved Context
# -------------------------------
print(f"User Query: {query}")
print("\n--- Retrieved Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"\nDocument {i}:\n{doc.page_content}")

# -------------------------------
# Prepare Prompt for LLM
# -------------------------------
combined_input = f"""
Based on the following documents, answer the question strictly using the provided information.

Question:
{query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

If the answer is not present, say:
"I don't have enough information to answer that question based on the provided documents."
"""

# -------------------------------
# Groq LLM
# -------------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",  # fast + cheap
    temperature=0
)

messages = [
    SystemMessage(content="You are a factual RAG assistant. Do not hallucinate."),
    HumanMessage(content=combined_input)
]

# -------------------------------
# Generate Response
# -------------------------------
response = llm.invoke(messages)

print("\n--- Generated Response ---")
print(response.content)
