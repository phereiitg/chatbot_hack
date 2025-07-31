from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List
import os
import uvicorn
from models.pdf_processor import PDFProcessor
from models.qa_engine import QAEngine

# Initialize the FastAPI app
app = FastAPI(
    title="PDF-QLM API",
    version="1.0.0",
    description="An API to answer questions from a PDF document using Google Gemini and LangChain.",
)

# --- Pydantic Models for Request and Response ---
class QuestionRequest(BaseModel):
    """Defines the structure for the incoming request body."""
    doc_url: str
    questions: List[str]

class QuestionResponse(BaseModel):
    """Defines the structure for the outgoing response body."""
    answers: List[str]

# --- Initialize Core Components ---
# These instances will be shared across API requests.
try:
    pdf_processor = PDFProcessor()
    qa_engine = QAEngine()
except Exception as e:
    # If initialization fails (e.g., missing API key), we'll catch it here.
    # The endpoint logic will then raise an appropriate HTTP exception.
    pdf_processor = None
    qa_engine = None
    initialization_error = e

# --- API Endpoints ---

@app.get("/", tags=["Status"])
async def root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "PDF-QLM API is running. Ready to answer questions from your documents!"}

@app.post("/api/v1/hackrx/run", response_model=QuestionResponse, tags=["Q&A"])
async def process_questions_from_pdf(request: QuestionRequest):
    """
    This endpoint processes a PDF from a URL, and answers a list of questions based on its content.
    """
    # Check if the core components initialized correctly
    if not pdf_processor or not qa_engine:
        raise HTTPException(status_code=500, detail=f"API Initialization Failed: {initialization_error}")

    try:
        # Step 1: Process the PDF from the provided URL to get document chunks.
        # This involves downloading, loading, and splitting the PDF text.
        print(f"Processing PDF from URL: {request.doc_url}")
        documents = await pdf_processor.process_pdf_from_url(request.doc_url)
        
        if not documents:
            raise HTTPException(status_code=400, detail="Failed to process the PDF document. It might be empty or corrupted.")

        # Step 2: Create a searchable vector store from the document chunks.
        # This converts the text into embeddings for semantic search.
        print("Creating vector store...")
        vector_store = await pdf_processor.create_vector_store(documents)
        
        # Step 3: Iterate through each question and get the answer from the QA engine.
        answers = []
        print(f"Processing {len(request.questions)} questions...")
        for question in request.questions:
            answer = await qa_engine.get_answer(vector_store, question)
            answers.append(answer)
        
        print("Successfully generated all answers.")
        return QuestionResponse(answers=answers)
    
    except HTTPException as http_exc:
        # Re-raise known HTTP exceptions
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors during processing.
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# --- Main entry point for running the app with Uvicorn ---
if __name__ == "__main__":
    # The port is determined by the environment variable `PORT`, which Railway sets automatically.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)