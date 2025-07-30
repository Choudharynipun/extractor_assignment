import cv2
import numpy as np
import pytesseract
import easyocr
import spacy
from spacy.training.example import Example
from spacy.util import minibatch, compounding
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
import logging
from pathlib import Path
import pandas as pd
from PIL import Image, ImageEnhance
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReceiptOCRProcessor:

    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def preprocess_image(self, image_path: str) -> np.ndarray:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        height, width = cleaned.shape
        if height < 800:
            scale_factor = 800 / height
            new_width = int(width * scale_factor)
            cleaned = cv2.resize(cleaned, (new_width, 800))

        return cleaned

    def extract_text_tesseract(self, image: np.ndarray) -> str:
        
        try:
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,/-:$#() '
            text = pytesseract.image_to_string(image, config=custom_config)
            return text.strip()
        except Exception as e:
            logger.warning(f"Tesseract OCR failed: {e}")
            return ""

    def extract_text_easyocr(self, image: np.ndarray) -> str:
        
        try:
            results = self.reader.readtext(image)
            text_lines = [result[1] for result in results if result[2] > 0.3]  # confidence > 0.3
            return '\n'.join(text_lines)
        except Exception as e:
            logger.warning(f"EasyOCR failed: {e}")
            return ""

    def extract_text(self, image_path: str) -> str:
      
        # Preprocess image
        processed_img = self.preprocess_image(image_path)

        # Try both OCR methods
        tesseract_text = self.extract_text_tesseract(processed_img)
        easyocr_text = self.extract_text_easyocr(processed_img)

        # Combine results (prefer the longer, more detailed one)
        if len(easyocr_text) > len(tesseract_text):
            primary_text = easyocr_text
            fallback_text = tesseract_text
        else:
            primary_text = tesseract_text
            fallback_text = easyocr_text

        # Combine unique information from both sources
        combined_text = primary_text
        if fallback_text:
            combined_text += "\n" + fallback_text

        return combined_text

class ReceiptNERModel:

    def __init__(self, model_path: Optional[str] = None):
        self.nlp = None
        if model_path and Path(model_path).exists():
            self.nlp = spacy.load(model_path)
        else:
            self.nlp = self._create_base_model()

    def _create_base_model(self):
      nlp = spacy.blank("en")  # ← This avoids the lookups error

     # Add NER component
      if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
      else:
        ner = nlp.get_pipe("ner")

     # Add custom labels
      ner.add_label("INVOICE_ID")
      ner.add_label("INVOICE_DATE")
      ner.add_label("LINE_ITEM")
      ner.add_label("QUANTITY")
      ner.add_label("PRICE")
      ner.add_label("TOTAL")

      return nlp


    def create_training_data(self) -> List[Tuple[str, Dict]]:

        training_data = [
            ("Invoice #12345 dated 2023-12-15", {
                "entities": [(8, 14, "INVOICE_ID"), (21, 31, "INVOICE_DATE")]
            }),
            ("Receipt ID: RCP-2023-001 Date: 01/15/2024", {
                "entities": [(12, 27, "INVOICE_ID"), (34, 44, "INVOICE_DATE")]
            }),
            ("Coffee Beans 2x $12.99 = $25.98", {
                "entities": [(0, 12, "LINE_ITEM"), (13, 15, "QUANTITY"), (16, 22, "PRICE"), (25, 31, "TOTAL")]
            }),
            ("Milk 1 bottle $3.50", {
                "entities": [(0, 4, "LINE_ITEM"), (5, 14, "QUANTITY"), (15, 20, "PRICE")]
            }),
            ("Order #ORD123456 on 2024-01-30", {
                "entities": [(6, 17, "INVOICE_ID"), (21, 31, "INVOICE_DATE")]
            }),
            ("Bread loaf qty:2 $4.99 each", {
                "entities": [(0, 10, "LINE_ITEM"), (15, 16, "QUANTITY"), (17, 22, "PRICE")]
            })
        ]
        return training_data

    def train_model(self, training_data: List[Tuple[str, Dict]], n_iter: int = 30) -> None:
        
        logger.info("Training NER model...")

        # Convert training data to spaCy format
        examples = []
        for text, annotations in training_data:
            doc = self.nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            examples.append(example)

        # Disable other pipes during training
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.begin_training()

            for i in range(n_iter):
                losses = {}
                # Batch the examples and iterate over them
                batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    self.nlp.update(batch, drop=0.5, losses=losses)

                if i % 10 == 0:
                    logger.info(f"Iteration {i}, Losses: {losses}")

        logger.info("Training completed!")

    def save_model(self, output_dir: str) -> None:
        """Save the trained model"""
        self.nlp.to_disk(output_dir)
        logger.info(f"Model saved to {output_dir}")

    def predict(self, text: str) -> Dict[str, List[str]]:
        
        doc = self.nlp(text)
        entities = {
            "INVOICE_ID": [],
            "INVOICE_DATE": [],
            "LINE_ITEM": [],
            "QUANTITY": [],
            "PRICE": [],
            "TOTAL": []
        }

        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text.strip())

        return entities

class ReceiptExtractor:

    def __init__(self, model_path: Optional[str] = None):
        self.ocr_processor = ReceiptOCRProcessor()
        self.ner_model = ReceiptNERModel(model_path)

        # Train with sample data if no pre-trained model
        if model_path is None or not Path(model_path).exists():
            training_data = self.ner_model.create_training_data()
            self.ner_model.train_model(training_data)

    def extract_with_regex(self, text: str) -> Dict[str, any]:
        
        results = {
            "invoice_id": [],
            "invoice_date": [],
            "line_items": []
        }

        # Invoice ID patterns
        invoice_patterns = [
            r'(?i)(?:invoice|receipt|order)[\s#:]*([A-Z0-9-]+)',
            r'(?i)#([A-Z0-9-]{3,})',
            r'(?i)(?:inv|rcp|ord)[\s#:-]*([A-Z0-9-]+)'
        ]

        for pattern in invoice_patterns:
            matches = re.findall(pattern, text)
            results["invoice_id"].extend(matches)

        # Date patterns
        date_patterns = [
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
            r'\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',
            r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4})\b'
        ]

        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            results["invoice_date"].extend(matches)

        # Line items with prices
        line_item_patterns = [
            r'([A-Za-z][A-Za-z\s]+)\s+.*?\$([0-9,]+\.?[0-9]*)',
            r'([A-Za-z][A-Za-z\s]+)\s+(\d+)\s*x?\s*\$([0-9,]+\.?[0-9]*)'
        ]

        for pattern in line_item_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) >= 2:
                    item_name = match[0].strip()
                    price = match[-1]
                    if len(item_name) > 2 and item_name.lower() not in ['total', 'tax', 'subtotal']:
                        results["line_items"].append({
                            "name": item_name,
                            "price": price
                        })

        return results

    def process_receipt(self, image_path: str) -> Dict[str, any]:
        
        logger.info(f"Processing receipt: {image_path}")

        try:
            # Step 1: Extract text using OCR
            extracted_text = self.ocr_processor.extract_text(image_path)

            if not extracted_text.strip():
                logger.warning("No text extracted from image")
                return {"error": "No text could be extracted from the image"}

            logger.info("OCR extraction completed")

            # Step 2: Extract entities using NER
            ner_results = self.ner_model.predict(extracted_text)
            print("\n[DEBUG] NER Results:\n", ner_results)


            # Step 3: Fallback to regex if NER doesn't find enough
            regex_results = self.extract_with_regex(extracted_text)
            print("\n[DEBUG] Regex Results:\n", regex_results)


            # Step 4: Combine and clean results
            final_results = {
                "extracted_text": extracted_text,
                "invoice_id": self._clean_and_dedupe(
                    ner_results.get("INVOICE_ID", []) + regex_results.get("invoice_id", [])
                ),
                "invoice_date": self._clean_and_dedupe(
                    ner_results.get("INVOICE_DATE", []) + regex_results.get("invoice_date", [])
                ),
                "line_items": self._extract_line_items(extracted_text, ner_results),
                "confidence_score": self._calculate_confidence(ner_results, regex_results)
            }

            logger.info("Receipt processing completed successfully")
            return final_results

        except Exception as e:
            logger.error(f"Error processing receipt: {e}")
            return {"error": str(e)}
        print("\n[DEBUG] Extracted OCR Text:\n", extracted_text)


    def _clean_and_dedupe(self, items: List[str]) -> List[str]:
        cleaned = []
        seen = set()

        for item in items:
            cleaned_item = re.sub(r'[^A-Za-z0-9-./]', '', item).strip()
            if cleaned_item and cleaned_item.lower() not in seen and len(cleaned_item) > 2:
                cleaned.append(cleaned_item)
                seen.add(cleaned_item.lower())

        return cleaned[:3]  # Return top 3 matches

    def _extract_line_items(self, text: str, ner_results: Dict) -> List[Dict]:
        
        line_items = []

        # Use NER results if available
        items = ner_results.get("LINE_ITEM", [])
        quantities = ner_results.get("QUANTITY", [])
        prices = ner_results.get("PRICE", [])

        # Combine NER results
        for i, item in enumerate(items):
            line_item = {"name": item.strip()}
            if i < len(quantities):
                line_item["quantity"] = quantities[i]
            if i < len(prices):
                line_item["price"] = prices[i]
            line_items.append(line_item)

        # Fallback to regex extraction
        if len(line_items) < 2:
            regex_pattern = r'([A-Za-z][A-Za-z\s]{2,15})\s+.*?\$([0-9,]+\.?[0-9]*)'
            matches = re.findall(regex_pattern, text)

            for match in matches:
                item_name = match[0].strip()
                price = match[1]

                # Filter out common non-item words
                skip_words = ['total', 'subtotal', 'tax', 'amount', 'change', 'cash', 'card']
                if not any(word in item_name.lower() for word in skip_words):
                    line_items.append({
                        "name": item_name,
                        "price": f"${price}"
                    })

        return line_items[:10]  # Return top 10 items

    def _calculate_confidence(self, ner_results: Dict, regex_results: Dict) -> float:

        score = 0.0
        total_possible = 3.0  # invoice_id, date, line_items

        if ner_results.get("INVOICE_ID") or regex_results.get("invoice_id"):
            score += 1.0
        if ner_results.get("INVOICE_DATE") or regex_results.get("invoice_date"):
            score += 1.0
        if ner_results.get("LINE_ITEM") or regex_results.get("line_items"):
            score += 1.0

        return round(score / total_possible, 2)

    def train_custom_model(self, training_data: List[Tuple[str, Dict]], model_save_path: str):
        
        self.ner_model.train_model(training_data, n_iter=50)
        self.ner_model.save_model(model_save_path)
        logger.info(f"Custom model trained and saved to {model_save_path}")

def main():
    
    # Initialize the extractor
    extractor = ReceiptExtractor()

    custom_training_data = [
        ("Store: Walmart Invoice: WMT123456 Date: 2024-01-15", {
            "entities": [(21, 30, "INVOICE_ID"), (37, 47, "INVOICE_DATE")]
        }),
        ("Apples 3 lbs $5.99 Bananas 2 lbs $3.49", {
            "entities": [(0, 6, "LINE_ITEM"), (7, 12, "QUANTITY"), (13, 18, "PRICE"),
                        (19, 26, "LINE_ITEM"), (27, 32, "QUANTITY"), (33, 38, "PRICE")]
        })
    ]


    print("Receipt OCR and NER System initialized successfully!")
    print("To use: extractor.process_receipt('path/to/receipt/image.jpg')")

if __name__ == "__main__":
    main()

def create_flask_api():
    """
    Optional Flask API wrapper for the receipt extractor
    """
    try:
        from flask import Flask, request, jsonify

        app = Flask(__name__)
        extractor = ReceiptExtractor()

        @app.route('/extract', methods=['POST'])
        def extract_receipt():
            if 'image' not in request.files:
                return jsonify({"error": "No image provided"}), 400

            file = request.files['image']
            if file.filename == '':
                return jsonify({"error": "No image selected"}), 400

            temp_path = f"/tmp/{file.filename}"
            file.save(temp_path)

            results = extractor.process_receipt(temp_path)

            Path(temp_path).unlink(missing_ok=True)

            return jsonify(results)

        return app

    except ImportError:
        logger.warning("Flask not installed. API endpoint not available.")
        return None
    
if __name__ == "__main__":
    import json

    extractor = ReceiptExtractor()
    results = extractor.process_receipt(r"C:\Users\choud\Downloads\archive (3)\SROIE2019\test\img\X51007846372.jpg")

    print(json.dumps(results, indent=2))
