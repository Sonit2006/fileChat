# engine.py (Updated for Hybrid: Local Ollama Embeddings + Azure OpenAI Chat)
# This script requires an internet connection for chat and Azure credentials.
# It also requires the Ollama application to be running for the embedding process.

import os
from langchain_community.vectorstores import FAISS
# New imports for Hybrid setup
from langchain_ollama import OllamaEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.docstore.document import Document

# --- Configuration ---
FAISS_INDEX_PATH = "./faiss_index"
DOCUMENTS_PATH = "./MyDrive"

# Use local Ollama for embeddings
EMBEDDING_MODEL_NAME = "nomic-embed-text"
# Use Azure OpenAI for chat
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")


def main():
    """
    Main function to run the RAG engine using a hybrid model setup.
    """
    print(f"--- Hybrid RAG Chatbot Engine (Embedding: {EMBEDDING_MODEL_NAME}) ---")
    
    # Check if Azure credentials for the chat model are set
    if not all([os.getenv("AZURE_OPENAI_API_KEY"), os.getenv("AZURE_OPENAI_ENDPOINT"), AZURE_CHAT_DEPLOYMENT]):
        print("\nFATAL ERROR: Azure OpenAI environment variables for the chat model are not set.")
        print("Please set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_CHAT_DEPLOYMENT_NAME.")
        return

    # Initialize local Ollama Embeddings
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)

    if os.path.exists(FAISS_INDEX_PATH):
        print(f"Loading existing FAISS index from {FAISS_INDEX_PATH}...")
        # You must use the same embedding model to load the index as was used to create it.
        db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"Index not found. Creating a new one at {FAISS_INDEX_PATH}...")
        # If you're creating a new index, you might need to delete the old one if the embedding model changed.
        db = create_new_db(embeddings)

    if not db:
        print("\nDatabase could not be created. Exiting.")
        return

    print("\nIndex loaded successfully.")
    
    retrieval_chain = create_rag_chain(db)
    
    # --- Interactive Chat Loop ---
    print("\nEnter your question to chat with your documents. Type 'exit' to quit.")
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'exit':
            break
        
        try:
            response = retrieval_chain.invoke({"input": question})
            
            print("\n--- Answer ---")
            print(response["answer"])
            
            print("\n--- Sources ---")
            if response["context"]:
                sources = set([doc.metadata['source'] for doc in response["context"]])
                for source in sources:
                    print(f"- {source}")
            else:
                print("No sources found for this answer.")
                
        except Exception as e:
            print(f"\nAn error occurred during question answering: {e}")

def load_db():
    """
    Loads the existing FAISS vector index if available, otherwise creates a new one.
    """
    print(f"Checking for existing FAISS index at {FAISS_INDEX_PATH}...")
    
    if os.path.exists(FAISS_INDEX_PATH):
        print("Found existing FAISS index. Loading...")
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
        db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("FAISS index loaded successfully.")
        return db
    else:
        print("No existing FAISS index found. Creating a new one...")
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
        return create_new_db(embeddings)
    
def create_new_db(embedding_function):
    """
    Creates a new FAISS vector index by loading text and indexing images.
    """
    print(f"Scanning and loading documents from {DOCUMENTS_PATH}...")
    
    all_docs = []
    text_extensions = {".pdf": PyPDFLoader, ".txt": TextLoader, ".docx": Docx2txtLoader}
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]

    for root, _, files in os.walk(DOCUMENTS_PATH):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_ext = os.path.splitext(filename)[1].lower()

            try:
                if file_ext in text_extensions:
                    loader_class = text_extensions[file_ext]
                    loader = loader_class(file_path)
                    loaded_docs = loader.load()
                    for doc in loaded_docs:
                        doc.page_content = f"Filename: {filename}\n\n{doc.page_content}"
                    all_docs.extend(loaded_docs)

                elif file_ext in image_extensions:
                    base_filename = os.path.splitext(filename)[0]
                    keywords_from_name = base_filename.replace('-', ' ').replace('_', ' ')
                    page_content = f"This is an image file. Keywords: image picture photo. The filename is {filename}. Keywords from filename: {keywords_from_name}."
                    dummy_doc = Document(page_content=page_content, metadata={"source": file_path})
                    all_docs.append(dummy_doc)

            except Exception as e:
                print(f"Error loading {filename}: {e}. Skipping.")

    if not all_docs:
        print("\nNo processable documents found.")
        return None

    print(f"\nLoaded/Indexed a total of {len(all_docs)} files.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)
    
    print(f"Split documents into {len(splits)} chunks.")
    print("Creating new FAISS index with local Ollama embeddings...")

    db = FAISS.from_documents(splits, embedding_function)
    
    db.save_local(FAISS_INDEX_PATH)
    
    print("FAISS index created and saved.")
    return db


def create_rag_chain(db):
    """
    Creates the complete RAG chain for question-answering.
    """
    # Initialize Azure OpenAI Chat Model
    model = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0,
)
    
    prompt = ChatPromptTemplate.from_template("""You are a helpful AI assistant that answers questions and finds files based on a collection of local documents.

Your main capabilities are:
1. Answering questions about the content of the documents.
2. Finding specific files by their name.

Analyze the user's question and the provided context to determine their intent.

- If the user's question is asking for information that can be found within the text of the documents (e.g., "what is the passport number of john?"), provide a direct answer based on the text content.
- If the user's question seems to be asking for a specific file (e.g., "where is the drivingLicense copy of jack rogers?"), search the 'source' metadata of the context documents. If you find a matching file, state that you have found it and provide the full file path.
- If the question is ambiguous, you can both answer the question based on the text and also mention any relevant files you found.

Always use the provided context to form your answer. If the information is not in the context, say that you cannot find the answer.

Question: {input} 

Context: 
{context} 

Answer:""")

    document_chain = create_stuff_documents_chain(model, prompt)
    
    retriever = db.as_retriever(search_kwargs={"k": 10})
    
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain


if __name__ == "__main__":
    # This setup requires the Ollama application to be running for embeddings.
    print("Make sure the Ollama application is running.")
    main()
