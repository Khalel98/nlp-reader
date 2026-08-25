from database import get_connection


def find_similar_chunks(
    document_id,
    question_embedding,
    top_k=3
):
    conn = get_connection()

    try:
        with conn.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    id,
                    content,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM chunks
                WHERE document_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (
                    question_embedding,
                    document_id,
                    question_embedding,
                    top_k
                )
            )

            rows = cursor.fetchall()

            return [
                {
                    "id": row[0],
                    "text": row[1],
                    "similarity": float(row[2])
                }
                for row in rows
            ]

    finally:
        conn.close()