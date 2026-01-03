import os
from PyPDF2 import PdfReader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "sanskrit_docs")

def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def load_documents(data_dir):
    documents = []

    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)

        if file.endswith(".txt"):
            documents.append(load_txt(file_path))

        elif file.endswith(".pdf"):
            documents.append(load_pdf(file_path))

    return documents

if __name__ == "__main__":
    docs = load_documents(DATA_DIR)
    print(f"Loaded {len(docs)} document(s)")
