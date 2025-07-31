from dotenv import load_dotenv
load_dotenv() # This line loads the environment variables from the .env file

import asyncio # Import asyncio for parallel processing
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict
import os
import uvicorn
from models.pdf_processor import PDFProcessor
from models.qa_engine import QAEngine
from langchain.vectorstores import FAISS

# --- CONFIGURATION ---
# Define the URL of the PDF you want to pre-process at startup.
# Paste your public PDF URL here. If you leave it empty, no pre-warming will occur.
PRE_WARM_DOC_URL = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"


# Initialize the FastAPI app
app = FastAPI(
    title="PDF-QLM API",
    version="1.3.0", # Version bump for pre-warming feature
    description="An API to answer questions from a PDF document using Google Gemini and LangChain, with in-memory caching and startup pre-warming.",
)

# --- In-Memory Cache for Vector Stores ---
vector_store_cache: Dict[str, FAISS] = {}

# --- Pydantic Models for Request and Response ---
class QuestionRequest(BaseModel):
    doc_url: str
    questions: List[str]

class QuestionResponse(BaseModel):
    answers: List[str]

# --- Initialize Core Components ---
try:
    pdf_processor = PDFProcessor()
    qa_engine = QAEngine()
except Exception as e:
    pdf_processor = None
    qa_engine = None
    initialization_error = e

# --- FastAPI Startup Event ---
@app.on_event("startup")
async def startup_event():
    """
    This event runs once when the application starts.
    It pre-processes a specified PDF to warm up the cache.
    """
    if pdf_processor and PRE_WARM_DOC_URL and PRE_WARM_DOC_URL not in vector_store_cache:
        print(f"STARTUP: Pre-warming cache for URL: {PRE_WARM_DOC_URL}")
        try:
            # This is the same logic as a cache miss
            documents = await pdf_processor.process_pdf_from_url(PRE_WARM_DOC_URL)
            if documents:
                vector_store = await pdf_processor.create_vector_store(documents)
                vector_store_cache[PRE_WARM_DOC_URL] = vector_store
                print(f"STARTUP CACHE POPULATED for: {PRE_WARM_DOC_URL}")
            else:
                print(f"STARTUP WARNING: Could not process pre-warm URL.")
        except Exception as e:
            print(f"STARTUP ERROR: Failed to pre-warm cache for {PRE_WARM_DOC_URL}. Error: {e}")


# --- API Endpoints ---
@app.get("/", tags=["Status"])
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "PDF-QLM API is running. Ready to answer questions from your documents!"}

@app.post("/api/v1/hackrx/run", response_model=QuestionResponse, tags=["Q&A"])
async def process_questions_from_pdf(request: QuestionRequest):
    """
    This endpoint processes a PDF from a URL, and answers a list of questions based on its content.
    It uses an in-memory cache for vector stores and processes questions in parallel.
    """
    if not pdf_processor or not qa_engine:
        raise HTTPException(status_code=500, detail=f"API Initialization Failed: {initialization_error}")

    doc_url = request.doc_url
    vector_store = None

    try:
        # Step 1: Check for the vector store in the cache first.
        if doc_url in vector_store_cache:
            print(f"CACHE HIT: Found vector store for URL: {doc_url}")
            vector_store = vector_store_cache[doc_url]
        else:
            # Step 2: If not in cache, process the PDF and create the vector store.
            print(f"CACHE MISS: Processing new PDF from URL: {doc_url}")
            documents = await pdf_processor.process_pdf_from_url(doc_url)
            
            if not documents:
                raise HTTPException(status_code=400, detail="Failed to process the PDF document. It might be empty or corrupted.")

            print("Creating new vector store...")
            vector_store = await pdf_processor.create_vector_store(documents)
            
            # Step 3: Save the newly created vector store to the cache.
            vector_store_cache[doc_url] = vector_store
            print(f"SAVED TO CACHE: Vector store for URL: {doc_url}")

        # Step 4: Process questions in parallel batches using the vector store.
        all_questions = request.questions
        batch_size = 50
        all_answers = []

        print(f"Processing {len(all_questions)} questions in batches of {batch_size}...")

        for i in range(0, len(all_questions), batch_size):
            batch_questions = all_questions[i:i + batch_size]
            tasks = [qa_engine.get_answer(vector_store, q) for q in batch_questions]
            batch_answers = await asyncio.gather(*tasks)
            all_answers.extend(batch_answers)
            print(f"Completed batch {i//batch_size + 1}...")
        
        print("Successfully generated all answers.")
        return QuestionResponse(answers=all_answers)
    
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# --- Main entry point for running the app with Uvicorn ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
