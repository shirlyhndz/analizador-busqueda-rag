from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import ollama

DB_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "llama3.2:latest"


def main():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    print("=== Analizador Inteligente de Problemas de Búsqueda ===")
    print("Escribe tu problema o consulta.\n")

    while True:
        query = input("Consulta: ")

        if query.lower() in ["salir", "exit"]:
            break

        results = vectorstore.similarity_search(query, k=3)

        context = "\n\n".join([doc.page_content for doc in results])

        prompt = f"""
Eres un asistente académico especializado en algoritmos de búsqueda en Inteligencia Artificial.

Debes responder únicamente usando el contexto proporcionado.

CONTEXTO:
{context}

CONSULTA:
{query}

INSTRUCCIONES:
- Recomienda el algoritmo más adecuado.
- Justifica técnicamente la respuesta.
- Explica ventajas y limitaciones.
- Si no hay suficiente contexto, indícalo.
"""

        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        print("\nRespuesta:\n")
        print(response["message"]["content"])
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()