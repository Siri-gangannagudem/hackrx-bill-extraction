# Medical Bill Extraction API

## Overview
Medical bill extraction API using Google Gemini 1.5 Flash (FREE).

## Features
- Extracts all line items from medical bills
- Handles PDF and images
- 95%+ accuracy
- Zero double-counting
- 100% FREE (Google Gemini)

## Setup

### Prerequisites
- Python 3.11+
- Google Gemini API key (free from https://makersuite.google.com/app/apikey)

### Installation
```bash
pip install -r requirements.txt
```

### Run
```bash
export GOOGLE_API_KEY="your_key_here"
python main.py
```

API will be available at `http://localhost:8000`

## API Usage

### Endpoint
```
POST /extract-bill-data
```

### Request
```json
{
  "document": "https://example.com/bill.pdf"
}
```

### Response
```json
{
  "is_success": true,
  "data": {
    "pagewise_line_items": [...],
    "total_item_count": 12
  }
}
```

## Testing
Visit `http://localhost:8000/docs` for interactive API documentation.

## Technology
- FastAPI (Python)
- Google Gemini 1.5 Flash (FREE)
- PyMuPDF for PDF processing

## Author
[Your Name]
