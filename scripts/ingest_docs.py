import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

DOCS_ROOT      = Path("./docs")
CHROMA_DIR     = "./chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"

# Map subfolder name → Chroma collection name
COLLECTION_MAP: dict[str, str] = {
    "cadquery":   "cadquery",
    # Add more: "my_api": "my_api"
}

CHUNK_SIZE    = 600
CHUNK_OVERLAP = 80

# ── Per-collection ingestion ──────────────────────────────────────────────────

def ingest_collection(folder: Path, collection_name: str) -> int:
    """Load all PDFs in a folder, chunk, embed, and store in a named collection."""
    print(f"\n📂 Ingesting '{folder.name}' → collection '{collection_name}'")

    loader = PyPDFDirectoryLoader(str(folder))
    documents = loader.load()

    if not documents:
        print(f"   ⚠️  No PDF files found in {folder}")
        return 0

    print(f"   📄 Loaded {len(documents)} pages from {len(set(d.metadata['source'] for d in documents))} file(s)")

    # Attach collection as metadata for filtering later
    for doc in documents:
        doc.metadata["collection"] = collection_name
        doc.metadata["source_folder"] = folder.name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"   ✂️  Split into {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=collection_name,
    )
    print(f"   ✅ Stored {len(chunks)} chunks in Chroma ['{collection_name}']")
    return len(chunks)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set in .env")

    total = 0
    found = False

    for folder_name, collection_name in COLLECTION_MAP.items():
        folder = DOCS_ROOT / folder_name
        if folder.exists() and any(folder.glob("**/*.pdf")):
            found = True
            total += ingest_collection(folder, collection_name)
        else:
            print(f"⏭️  Skipping '{folder_name}' — no PDFs found")

    if not found:
        print("\n❌ No PDFs found in any docs/ subfolder. Add PDFs and retry.")
        return

    print(f"\n🎉 Ingestion complete — {total} total chunks stored in '{CHROMA_DIR}'")


if __name__ == "__main__":
    main()