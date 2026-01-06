# api_server.py
# This script runs a web server to expose your RAG engine and serve files.

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cloud_engine as engine  # Imports the engine.py script
import os
import pathlib
import mimetypes

app = FastAPI(
    title="Local RAG API Server",
    description="An API to chat with your local documents and serve files.",
)

# Allow CORS for the web app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

print("Loading the RAG engine... This may take a moment.")
db = engine.load_db()
rag_chain = engine.create_rag_chain(db)
print("RAG engine loaded successfully. Server is ready.")

# API Endpoints 
@app.get("/")
async def read_root():
    """ Serves the main index.html file. """
    try:
        with open("index.html") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found")

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """ Receives a question, gets an answer, and returns it. """
    try:
        print(f"Received question: {request.question}")
        response = rag_chain.invoke({"input": request.question})
        
        answer_data = {
            "question": request.question,
            "answer": response.get("answer", "No answer found."),
            "sources": list(set([doc.metadata.get('source', 'Unknown') for doc in response.get("context", [])]))
        }
        print(f"Sending answer: {answer_data['answer']}")
        return answer_data
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/{file_path:path}")
async def get_file(file_path: str):
    """ Safely serves a file from the documents directory. """
    try:
        requested_path = pathlib.Path(file_path).resolve()

        if not requested_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

            
        media_type, _ = mimetypes.guess_type(requested_path)
        return FileResponse(requested_path, media_type=media_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
