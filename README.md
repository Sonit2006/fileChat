# filepath: /Users/sonit/Documents/fileChat/README.md
# FileChat - Document RAG Example

This repository demonstrates a hybrid RAG (Retrieval-Augmented Generation) setup using local Ollama embeddings and Azure OpenAI for chat.

Project description

This is a RAG (Retrieval-Augmented Generation) application that scans files and folders, builds a FAISS index of document chunks using vector embeddings, and allows fast semantic search and chat over large local document collections. The index uses approximate nearest neighbors (ANN) for efficient similarity search. The typical workflow is:

- Index: load documents, split them into chunks, compute vector embeddings, and store them in a FAISS index.
- Retrieve: given a user question, retrieve the most relevant chunks using ANN similarity search.
- Generate: pass retrieved context to a chat model to produce a concise answer, optionally returning source file paths.

This project uses nomic-embed-text (via the local Ollama embeddings) to compute vector representations and Azure OpenAI (chat) to generate responses.

Key capabilities

- Build a FAISS index from a folder of documents for fast similarity search (ANN).
- Chat with files to quickly extract information from large folders.
- Return source file paths for traceability.

Tech stack and models

- Language: Python
- Vector store: FAISS (local)
- Embeddings: nomic-embed-text (local via Ollama)
- Chat / response generation: Azure OpenAI (chat deployment)
- Orchestration / helper libraries: LangChain and related extensions
- API: FastAPI (lightweight server example)
- UI: Streamlit (simple demo)

Files and structure
.
├── app/
│   └── streamlit_app.py
├── core/
│   ├── engine.py
│   ├── cloud_engine.py
│   └── api_server.py
├── indexing/
│   └── build_index.py
├── sample_docs/
│   └── example.txt
├── .env.example
├── requirements.txt
├── .gitignore
└── README.md

Getting started

1. Copy .env.example to .env and fill in your Azure credentials. Do NOT commit .env.
2. Install dependencies: pip install -r requirements.txt
3. Ensure Ollama is running locally if you use local embeddings.
4. Build the index: python indexing/build_index.py
5. Run the API: uvicorn core.api_server:app --reload
6. Run the streamlit demo: streamlit run app/streamlit_app.py

Security

- This repo uses environment variables for secrets.
- FAISS index and documents are ignored by .gitignore.
