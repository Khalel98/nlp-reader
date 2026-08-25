from google import genai


def create_embedding(client, text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config={
            "output_dimensionality": 768
        }
    )

    return response.embeddings[0].values


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
    )

    embedding = create_embedding(
        client,
        "Договор действует три года."
    )

    print("Размер embedding:", len(embedding))
    print("Первые 10 чисел:", embedding[:10])