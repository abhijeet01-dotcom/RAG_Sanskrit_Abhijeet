from retriever import retrieve
from generator import generate_answer

def run_rag():
    print("Sanskrit RAG System (CPU-only)")
    query = input("Enter your query (Sanskrit or transliteration): ")

    contexts = retrieve(query, top_k=3)
    combined_context = "\n".join(contexts)

    answer = generate_answer(combined_context, query)

    print("\n Answer:")
    print(answer)

if __name__ == "__main__":
    run_rag()
