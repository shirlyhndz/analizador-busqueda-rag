from pathlib import Path
import fitz  # PyMuPDF

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


DATA_DIR = Path("data")
DB_DIR = Path("chroma_db")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def read_pdf(file_path: Path) -> str:
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text() + "\n"
    return text


def read_txt(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="ignore")


def load_documents() -> list[dict]:
    documents = []

    for file_path in DATA_DIR.glob("*"):
        if file_path.suffix.lower() == ".pdf":
            content = read_pdf(file_path)
        elif file_path.suffix.lower() == ".txt":
            content = read_txt(file_path)
        else:
            continue

        if content.strip():
            documents.append({
                "content": content,
                "source": file_path.name
            })

    return documents


def main() -> None:
    if not DATA_DIR.exists():
        print("No existe la carpeta data.")
        return

    documents = load_documents()

    if not documents:
        print("No hay documentos PDF o TXT en la carpeta data.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    texts = []
    metadatas = []

    for doc in documents:
        chunks = splitter.split_text(doc["content"])

        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append({
                "source": doc["source"],
                "chunk": i
            })

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=str(DB_DIR)
    )

    print("Indexación finalizada correctamente.")
    print(f"Documentos procesados: {len(documents)}")
    print(f"Fragmentos creados: {len(texts)}")
    print(f"Base vectorial guardada en: {DB_DIR}")


if __name__ == "__main__":
    main()