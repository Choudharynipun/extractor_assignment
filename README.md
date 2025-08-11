### OCR SYSTEM FOR RECIEPT PROCESSING, FOR SMALL SCALE ENTERPRISES:- 

This is an OCR_NER based system used to process the data in Reciepts for better processing and analysis, made for small scale enterprises.
It has the following key features: -

* **Modular Design**
  The entire pipeline is broken into clean modules — OCR engine, NER model, and Regex matcher — making it easy to debug, extend, or swap components.

* **Regex + NER Combination**
  To improve accuracy, the system combines spaCy’s NER with regex-based extraction. This means even if one method fails, the other can still extract the data correctly.

* **Confidence Scores**
  The output includes confidence scores for each prediction, giving us better transparency and control over the extraction reliability.

* **Image Preprocessing for Better OCR**
  Invoice images go through denoising, binarization, and resizing to improve OCR performance — especially helpful when dealing with low-quality scans.

* **Scalable Field Extraction**
  The project is designed to be extended easily. We can add more entities like `GSTIN`, `Vendor Name`, etc., just by updating the labels and training data.

* **API Ready (Flask)**
  Comes with a simple Flask API (`/extract`) where we can upload an image and get back structured invoice data which can be plugged into any web or mobile app


