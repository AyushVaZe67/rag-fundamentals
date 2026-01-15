from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# -------------------------------
# Connect to Chroma DB
# -------------------------------
persistent_directory = "db/chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"}
)

# -------------------------------
# Groq LLM
# -------------------------------
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# -------------------------------
# Conversation Memory
# -------------------------------
chat_history = []

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    # ------------------------------------
    # Step 1: Rewrite follow-up question
    # ------------------------------------
    if chat_history:
        rewrite_messages = [
            SystemMessage(
                content=(
                    "Given the chat history, rewrite the new question so that it is "
                    "a standalone, clear, and searchable question. "
                    "Return ONLY the rewritten question."
                )
            ),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        rewrite_result = model.invoke(rewrite_messages)
        search_question = rewrite_result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # ------------------------------------
    # Step 2: Retrieve documents
    # ------------------------------------
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        preview = "\n".join(doc.page_content.split("\n")[:2])
        print(f"  Doc {i}: {preview}...")

    # ------------------------------------
    # Step 3: Build grounded prompt
    # ------------------------------------
    combined_input = f"""
Based on the following documents, answer the user's question strictly using the provided information.

Question:
{user_question}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in docs])}

If the answer is not present in the documents, say:
"I don't have enough information to answer that question based on the provided documents."
"""

    # ------------------------------------
    # Step 4: Generate answer
    # ------------------------------------
    answer_messages = [
        SystemMessage(
            content="You are a factual RAG assistant. Do not hallucinate."
        ),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(answer_messages)
    answer = result.content.strip()

    # ------------------------------------
    # Step 5: Store conversation
    # ------------------------------------
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")
    return answer


# -------------------------------
# Chat Loop
# -------------------------------
def start_chat():
    print("Ask me questions! Type 'quit' to exit.")

    while True:
        question = input("\nYour question: ")

        if question.lower() == "quit":
            print("Goodbye!")
            break

        ask_question(question)


if __name__ == "__main__":
    start_chat()
