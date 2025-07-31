import requests
import tempfile
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from typing import List

class PDFProcessor:
    """
    Handles downloading, loading, splitting, and embedding PDF documents.
    """
    def __init__(self):
        """Initializes the PDF processor with embeddings and a text splitter."""
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not found.")
            
        # Initialize Google Gemini embeddings model
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Size of each text chunk
            chunk_overlap=200 # Overlap between chunks to maintain context
        )
    
    async def process_pdf_from_url(self, pdf_url: str) -> List:
        """
        Downloads a PDF from a URL, loads it, and splits it into documents.
        """
        try:
            # Download the PDF content from the URL
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            
            # Use a temporary file to store the PDF content
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name
            
            # Load the PDF from the temporary file path
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            
            # Split the loaded documents into smaller chunks for processing
            splits = self.text_splitter.split_documents(documents)
            return splits
        
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to download PDF from URL: {pdf_url}. Error: {e}")
        finally:
            # Clean up the temporary file after processing
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
    
    async def create_vector_store(self, documents: List):
        """
        Creates a FAISS vector store from a list of document chunks.
        """
        if not documents:
            return None
        try:
            # Create the vector store using the document splits and Gemini embeddings
            vector_store = FAISS.from_documents(documents, self.embeddings)
            return vector_store
        except Exception as e:
            raise RuntimeError(f"Failed to create FAISS vector store. Error: {e}")