## Sanskrit Document Retrieval-Augmented Generation (RAG) System

## Project Overview :-
This project implements a CPU-based RAG system for Sanskrit documents.
It retrieves relevant Sanskrit text from documents and generates answers using a LLM model

## Features :-
- Supports Sanskrit text documents
- Unicode-safe text handling
- Semantic retrieval using FAISS
- CPU-only inference
- Modular RAG architecture

## Project Structure
RAG_Sanskrit_Abhijeet/
- code/
- data/
- report/
- README.md

## How to Run

Step 1: Create vector index
python code/embed_store.py

Step 2: Run the system
python code/main.py

## Models Used
- Embedding Model: Multilingual Sentence Transformer
- Vector Store: FAISS
- LLM: FLAN-T5 (CPU)

## Notes
The Sanskrit documents used were provided as part of the assignment.

## Author
Abhijeet Kumar
