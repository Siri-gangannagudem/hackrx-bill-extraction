from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import requests
from typing import List, Optional
import json
import os
from io import BytesIO
from PIL import Image
import PyPDF2

app = FastAPI()

# Initialize Google Gemini (FREE API with generous limits)
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

class DocumentRequest(BaseModel):
    document: str

class BillItem(BaseModel):
    item_name: str
    item_amount: float
    item_rate: float
    item_quantity: float

class PagewiseLineItems(BaseModel):
    page_no: str
    page_type: Optional[str] = None
    bill_items: List[BillItem]

class TokenUsage(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int

class ExtractionData(BaseModel):
    pagewise_line_items: List[PagewiseLineItems]
    total_item_count: int

class ExtractionResponse(BaseModel):
    is_success: bool
    token_usage: TokenUsage
    data: ExtractionData

def download_document(url: str) -> bytes:
    """Download document from URL"""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content

def is_pdf(document_bytes: bytes) -> bool:
    """Check if document is PDF"""
    return document_bytes[:4] == b'%PDF'

def pdf_to_images(pdf_bytes: bytes) -> List[Image.Image]:
    """Convert PDF pages to images using PyMuPDF (fitz)"""
    import fitz  # PyMuPDF
    
    images = []
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        # Render page to image at 2x resolution for better quality
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    
    pdf_document.close()
    return images

def extract_bill_data_with_gemini(document_bytes: bytes) -> dict:
    """Extract bill data using Google Gemini (FREE)"""
    
    # Initialize Gemini Pro Vision model (FREE with 60 requests/minute)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Prepare images
    images = []
    if is_pdf(document_bytes):
        # Convert PDF to images
        images = pdf_to_images(document_bytes)
    else:
        # Single image
        image = Image.open(BytesIO(document_bytes))
        images = [image]
    
    # Enhanced prompt for accurate extraction
    extraction_prompt = """You are an expert medical billing data extraction system. Analyze this medical bill/invoice image and extract ALL line items with 100% accuracy.

EXTRACTION RULES:
1. Extract EVERY single line item - missing items is a critical error
2. NEVER double-count items - each item should appear exactly once
3. Extract exact values as shown in the document
4. For each item, extract:
   - item_name: Exact text as written (preserve formatting)
   - item_amount: Net amount (after discounts/adjustments) 
   - item_rate: Price per unit
   - item_quantity: Number of units

HANDLING EDGE CASES:
- If quantity not shown: assume 1.0
- If rate not shown: rate = amount
- If unclear, set to 0.0 but extract the item
- Include sub-totals only if they appear as distinct line items

PAGE CLASSIFICATION:
Determine page_type:
- "Bill Detail": Itemized services/procedures
- "Final Bill": Summary with totals
- "Pharmacy": Medication/drug items

CRITICAL: Return ONLY valid JSON in this EXACT format (no markdown, no extra text):
{
    "pagewise_line_items": [
        {
            "page_no": "1",
            "page_type": "Bill Detail",
            "bill_items": [
                {
                    "item_name": "Consultation",
                    "item_amount": 1000.00,
                    "item_rate": 1000.00,
                    "item_quantity": 1.00
                }
            ]
        }
    ],
    "total_item_count": 1
}

Extract ALL items from the bill now."""

    # Process each page/image
    all_items = []
    total_input_tokens = 0
    total_output_tokens = 0
    
    for page_num, image in enumerate(images, start=1):
        try:
            # Generate content with image
            response = model.generate_content(
                [extraction_prompt, image],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=4096,
                )
            )
            
            # Track token usage (approximate for Gemini)
            total_input_tokens += len(extraction_prompt.split()) * 1.3  # Approximate
            total_output_tokens += len(response.text.split()) * 1.3
            
            # Extract and parse JSON from response
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON
            page_data = json.loads(response_text)
            
            # Update page numbers
            if "pagewise_line_items" in page_data:
                for page_item in page_data["pagewise_line_items"]:
                    page_item["page_no"] = str(page_num)
                    all_items.extend(page_item.get("bill_items", []))
            
        except json.JSONDecodeError as e:
            print(f"JSON parse error on page {page_num}: {e}")
            print(f"Response text: {response_text[:200]}")
            # Try to extract items even from malformed JSON
            continue
        except Exception as e:
            print(f"Error processing page {page_num}: {e}")
            continue
    
    # Combine all pages into final result
    extracted_data = {
        "pagewise_line_items": [
            {
                "page_no": "1",
                "page_type": "Bill Detail",
                "bill_items": all_items
            }
        ],
        "total_item_count": len(all_items)
    }
    
    # Token usage
    token_usage = {
        "total_tokens": int(total_input_tokens + total_output_tokens),
        "input_tokens": int(total_input_tokens),
        "output_tokens": int(total_output_tokens)
    }
    
    return extracted_data, token_usage

@app.post("/extract-bill-data", response_model=ExtractionResponse)
async def extract_bill_data(request: DocumentRequest):
    """
    Extract line items and totals from medical bills/invoices using FREE Google Gemini
    """
    try:
        # Download document
        document_bytes = download_document(request.document)
        
        # Extract data using Gemini (FREE)
        extracted_data, token_usage = extract_bill_data_with_gemini(document_bytes)
        
        # Validate and construct response
        response = {
            "is_success": True,
            "token_usage": token_usage,
            "data": extracted_data
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )

@app.get("/")
async def root():
    return {
        "message": "Medical Bill Extraction API (FREE - Google Gemini)", 
        "status": "running",
        "model": "gemini-1.5-flash (FREE)"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "gemini-1.5-flash"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)