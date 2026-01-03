import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "vector.index")
CHUNKS_PATH = os.path.join(BASE_DIR, "chunks.pkl")

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

def load_vector_store():
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def retrieve(query, top_k=3):
    index, chunks = load_vector_store()
    query_embedding = model.encode([query])
    _, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

if __name__ == "__main__":
    query = input("Enter a query: ")
    results = retrieve(query)
    for r in results:
        print(r)
