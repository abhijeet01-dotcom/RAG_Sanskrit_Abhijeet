import re

def clean_text(text):
    """
    Cleans Sanskrit text by removing extra whitespace.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text, chunk_size=400, overlap=50):
    """
    Splits text into overlapping chunks.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

