import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from PyPDF2 import PdfReader
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# T5-based model (Flan-T5)
qa_model = pipeline("text2text-generation", model="google/flan-t5-base")

# (ID -> PDF text)
pdf_storage = {}

# Upload a PDF file to extract its text
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}")
        # Read and extract
        pdf_reader = PdfReader(file.file)
        pdf_text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            pdf_text += page.extract_text()
        
        if not pdf_text.strip():
            logger.warning(f"No text found in the uploaded PDF: {file.filename}")
            raise HTTPException(status_code=400, detail="The PDF contains no readable text.")
        
        pdf_id = file.filename
        
        # Store the text in memory
        pdf_storage[pdf_id] = pdf_text
        logger.info(f"PDF processed and stored with id: {pdf_id}")
        
        return {
            "message": "PDF uploaded successfully!",
            "pdf_id": pdf_id,
            "instructions": "You can ask questions about this PDF using /chat_about_pdf/{pdf_id} endpoint."
        }
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the PDF.")

# Chat with the PDF
@app.post("/chat_about_pdf/{pdf_id}")
async def chat_about_pdf(pdf_id: str, question: str):
    try:
        # Check if the PDF's text exists
        if pdf_id not in pdf_storage:
            logger.warning(f"PDF with id {pdf_id} not found.")
            raise HTTPException(status_code=404, detail="PDF not found.")
        
        pdf_text = pdf_storage[pdf_id]
        logger.info(f"Asking question on PDF id: {pdf_id}")
        
        # Get an answer to the question (using the Flan-T5 model)
        answer = qa_model(f"question: {question} context: {pdf_text}")
        
        return {"answer": answer[0]['generated_text']}
    except Exception as e:
        logger.error(f"Error during question-answer process: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your question.")
