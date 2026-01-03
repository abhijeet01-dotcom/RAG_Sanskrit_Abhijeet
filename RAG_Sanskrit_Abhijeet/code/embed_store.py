import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

from load_docs import load_documents
from preprocess import clean_text, chunk_text

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "sanskrit_docs")
INDEX_PATH = os.path.join(BASE_DIR, "vector.index")
CHUNKS_PATH = os.path.join(BASE_DIR, "chunks.pkl")

def create_vector_store():
    # 1. Load documents
    documents = load_documents(DATA_DIR)

    all_chunks = []

    for doc in documents:
        clean = clean_text(doc)
        chunks = chunk_text(clean)
        all_chunks.extend(chunks)

    print(f"Total text chunks created: {len(all_chunks)}")

    # 2. Load multilingual embedding model (CPU-friendly)
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 3. Generate embeddings
    embeddings = model.encode(all_chunks, show_progress_bar=True)

    # 4. Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # 5. Save index and chunks
    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print("Vector store created successfully")
    print(f"FAISS index size: {index.ntotal}")

if __name__ == "__main__":
    create_vector_store()
