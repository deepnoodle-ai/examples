import os
import numpy as np
from openai import OpenAI
from typing import List, Tuple

# Initialize the OpenAI client
client = OpenAI(
    api_key=os.getenv(
        "OPENAI_API_KEY"
    )  # Make sure to set your API key as an environment variable
)

# Sample texts for demonstration
sample_texts = [
    "The quick brown fox jumps over the lazy dog",
    "A fast auburn canine leaps above a sleepy hound",
    "Python is a versatile programming language",
    "JavaScript is widely used for web development",
    "Machine learning enables computers to learn from data",
    "Artificial intelligence is transforming technology",
    "The weather is sunny and warm today",
    "It's a beautiful day with clear skies",
]


def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """
    Get the embedding for a given text using OpenAI's embedding model.

    Args:
        text: The text to embed
        model: The embedding model to use (default: text-embedding-3-small)

    Returns:
        List of float values representing the embedding
    """
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate the cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score between -1 and 1
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    return dot_product / (norm1 * norm2)


def find_most_similar(
    query: str, texts: List[str], embeddings: List[List[float]], top_k: int = 3
) -> List[Tuple[str, float]]:
    """
    Find the most similar texts to a query based on embedding similarity.

    Args:
        query: The query text
        texts: List of texts to search through
        embeddings: Pre-computed embeddings for the texts
        top_k: Number of most similar results to return

    Returns:
        List of tuples containing (text, similarity_score)
    """
    # Get embedding for the query
    print(f"Getting embedding for query: '{query}'")
    query_embedding = get_embedding(query)

    # Calculate similarities
    similarities = []
    for i, text in enumerate(texts):
        similarity = cosine_similarity(query_embedding, embeddings[i])
        similarities.append((text, similarity))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]


def main():
    """Main function to demonstrate similarity search with OpenAI embeddings."""

    print("OpenAI Embeddings Similarity Search Demo")
    print("=" * 50)

    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set the OPENAI_API_KEY environment variable")
        print("You can do this by running: export OPENAI_API_KEY='your-api-key'")
        return

    # Generate embeddings for all sample texts
    print("\nGenerating embeddings for sample texts...")
    embeddings = []
    for text in sample_texts:
        print(f"  - {text[:50]}...")
        embedding = get_embedding(text)
        embeddings.append(embedding)

    print(f"\nGenerated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")

    # Demonstrate similarity search with different queries
    queries = [
        "A dog jumping",
        "Programming languages",
        "AI and ML",
        "Nice weather outside",
    ]

    print("\n" + "=" * 50)
    print("SIMILARITY SEARCH RESULTS")
    print("=" * 50)

    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)

        results = find_most_similar(query, sample_texts, embeddings, top_k=3)

        for i, (text, similarity) in enumerate(results, 1):
            print(f"{i}. {text}")
            print(f"   Similarity: {similarity:.4f}")

    # Interactive search
    print("\n" + "=" * 50)
    print("INTERACTIVE SEARCH")
    print("=" * 50)
    print("Enter your own queries (type 'quit' to exit):\n")

    while True:
        user_query = input("Your query: ").strip()
        if user_query.lower() in ["quit", "exit", "q"]:
            break

        if user_query:
            results = find_most_similar(user_query, sample_texts, embeddings, top_k=3)
            print("\nTop 3 similar texts:")
            for i, (text, similarity) in enumerate(results, 1):
                print(f"{i}. {text}")
                print(f"   Similarity: {similarity:.4f}")
            print()


if __name__ == "__main__":
    main()
