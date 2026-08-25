import numpy as np


def cosine_similarity(vector_a, vector_b):
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)

    similarity = np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )

    return similarity


def find_similar_chunks(
    question_embedding,
    document_chunks,
    top_k=3
):
    results = []

    for chunk in document_chunks:
        similarity = cosine_similarity(
            question_embedding,
            chunk["embedding"]
        )

        results.append({
            "text": chunk["text"],
            "similarity": similarity
        })

    results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return results[:top_k]