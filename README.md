A scalable system for extracting invoice information from receipt images using OCR and Named Entity Recognition (NER).

## Features

- **Dual OCR Engine**: Uses both Tesseract and EasyOCR for maximum text extraction accuracy
- **Custom NER Model**: SpaCy-based Named Entity Recognition for structured data extraction
- **Scalable Architecture**: Easy to extend for additional field types
- **High Accuracy**: Combines NER and regex fallback for robust extraction

## Extracted Information

- âœ… **Invoice Number/ID**
- âœ… **Invoice Date** 
- âœ… **Line Items** (product names, quantities, prices)
- âœ… **Confidence Scores**

## Installation

### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
- Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Quick Start

```python
from receipt_ocr_ner_system import ReceiptExtractor
import json

# Initialize the system
extractor = ReceiptExtractor()

# Process a receipt image
results = extractor.process_receipt(r"C:\Users\choud\Downloads\archive (3)\SROIE2019\test\img\X51007846372.jpg")

# Print extracted information
print(json.dumps(results, indent=2))
```

## Example Output

```json
{
  "extracted_text": "Store Name
Invoice #12345
Date: 2024-01-15
Coffee $4.99
Bread $2.50
Total: $7.49",
  "invoice_id": ["12345"],
  "invoice_date": ["2024-01-15"],
  "line_items": [
    {
      "name": "Coffee",
      "price": "$4.99"
    },
    {
      "name": "Bread", 
      "price": "$2.50"
    }
  ],
  "confidence_score": 1.0
}
```

## Advanced Usage

### Training Custom Models

```python
# Define custom training data
training_data = [
    ("Store: ABC Market Invoice: ABC123 Date: 2024-01-15", {
        "entities": [(22, 28, "INVOICE_ID"), (35, 45, "INVOICE_DATE")]
    }),
    ("Milk 2x $3.99 = $7.98", {
        "entities": [(0, 4, "LINE_ITEM"), (5, 7, "QUANTITY"), (8, 13, "PRICE")]
    })
]

# Train and save custom model
extractor.train_custom_model(training_data, "my_custom_model")

# Load custom model
custom_extractor = ReceiptExtractor("my_custom_model")
```

### API Integration

```python
from receipt_ocr_ner_system import create_flask_api

app = create_flask_api()

if __name__ == '__main__':
    app.run(debug=True)
```

## API Endpoint

**POST /extract**
- Upload image file
- Returns extracted receipt data as JSON

```bash
curl -X POST -F "image=@receipt.jpg" http://localhost:5000/extract
```

## Architecture

### Core Components

1. **ReceiptOCRProcessor**: Handles image preprocessing and text extraction
2. **ReceiptNERModel**: Custom NER model for entity recognition
3. **ReceiptExtractor**: Main orchestrator combining OCR and NER

### Processing Pipeline

1. **Image Preprocessing**: Noise reduction, thresholding, resizing
2. **OCR Extraction**: Dual-engine text extraction (Tesseract + EasyOCR)
3. **NER Processing**: Custom entity recognition
4. **Regex Fallback**: Pattern matching for missed entities
5. **Post-processing**: Cleaning, deduplication, confidence scoring

## Customization

### Adding New Entity Types

```python
# In ReceiptNERModel._create_base_model()
ner.add_label("STORE_NAME")
ner.add_label("TAX_AMOUNT")
ner.add_label("DISCOUNT")

# Add training data for new entities
training_data = [
    ("Store: Walmart Tax: $2.50", {
        "entities": [(7, 14, "STORE_NAME"), (20, 25, "TAX_AMOUNT")]
    })
]
```

### Custom Regex Patterns

```python
# In ReceiptExtractor.extract_with_regex()
store_patterns = [
    r'(?i)store[\s:]*([A-Za-z\s]+)',
    r'(?i)([A-Za-z\s]+)\s+store'
]
```

## Performance Tips

1. **Image Quality**: Higher resolution images yield better OCR results
2. **Preprocessing**: Clean, high-contrast images work best
3. **Training Data**: More diverse training data improves NER accuracy
4. **Model Tuning**: Adjust confidence thresholds based on your data

## Error Handling

The system includes comprehensive error handling:
- Invalid image files
- OCR failures
- Missing dependencies
- Model loading errors
