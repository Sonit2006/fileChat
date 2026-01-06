# filepath: /Users/sonit/Documents/fileChat/indexing/build_index.py
"""
Simple script to build the FAISS index from DOCUMENTS_PATH using the core cloud engine.
"""

from core.cloud_engine import create_new_db, OllamaEmbeddings

if __name__ == "__main__":
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = create_new_db(embeddings)
    if db:
        print("Index built successfully.")
    else:
        print("Index build failed.")
