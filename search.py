from database import get_connection


def find_similar_chunks(
    document_id,
    question_embedding,
    top_k=3
):
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
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
                    "text": row[0],
                    "similarity": float(row[1])
                }
                for row in rows
            ]

    finally:
        connection.close()