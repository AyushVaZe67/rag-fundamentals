from langchain_text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

tesla_text = """Tesla's Q3 Results

Tesla reported record revenue of $25.2B in Q3 2024.

Model Y Performance

The Model Y became the best-selling vehicle globally, with 350,000 units sold.

Production Challenges

Supply chain issues caused a 12% increase in production costs.

This is one very long paragraph that definitely exceeds our 100 character limit and has no double newlines inside it whatsoever making it impossible to split properly."""


splitter1 = CharacterTextSplitter(
    # separator=" ",  # Default separator. Other options include ["\n\n", "\n", ". ", " ", ""]
    chunk_size=100,
    chunk_overlap=0
)