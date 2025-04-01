from flask import Flask, render_template, request, send_file
import os
import requests
import torch
from PIL import Image
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from googlesearch import search
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from pdfminer.high_level import extract_text
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import cv2
import numpy as np
import pytesseract
# Manually set the Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from collections import Counter
import re
import fitz  # PyMuPDF for PDF image extraction
from io import BytesIO
import transformers
transformers.logging.set_verbosity_error()

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
REPORT_FOLDER = "reports"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# Load AI detector model
device = "cuda" if torch.cuda.is_available() else "cpu"
ai_model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector").to(device)
tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")

def read_file(file_path):
    ext = file_path.split(".")[-1].lower()
    if ext == "pdf":
        return extract_text_with_images_from_pdf(file_path)
    elif ext == "docx":
        return extract_text_with_images_from_docx(file_path)
    elif ext == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif ext in ["png", "jpg", "jpeg"]:
        return extract_text_from_image(file_path)
    return ""

def extract_text_from_image(image_path):
    """Extract text from images using Tesseract OCR."""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text.strip()

def preprocess_handwritten_image(img):
    """Preprocess the image to enhance handwritten OCR accuracy."""
    img = np.array(img.convert("L"))  # Convert to grayscale
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # Upscale to improve clarity
    img = cv2.GaussianBlur(img, (5, 5), 0)  # Reduce noise
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)  # Adaptive binarization
    return Image.fromarray(img)

def extract_text_with_images_from_pdf(pdf_path):
    """Extract handwritten text from images in a PDF using OCR."""
    doc = fitz.open(pdf_path)
    extracted_text = ""

    for page in doc:
        # Extract printed text normally
        extracted_text += page.get_text("text") + "\n"

        # Extract images and apply OCR
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_data = base_image["image"]
            img = Image.open(BytesIO(img_data))

            # Preprocess the image for handwritten OCR
            img = preprocess_handwritten_image(img)

            # Perform OCR using an optimized PSM
            text = pytesseract.image_to_string(img, config="--psm 4")  # Try --psm 4 or --psm 11 for handwritten text
            extracted_text += "\n" + text

    return extracted_text.strip()

def extract_text_with_images_from_docx(docx_path):
    """Extract text from both text content and images in a DOCX file."""
    doc = Document(docx_path)
    text = "\n".join([p.text for p in doc.paragraphs])

    # Extract images and perform OCR
    for rel in doc.part.rels:
        if "image" in doc.part.rels[rel].target_ref:
            image = doc.part.rels[rel].target_part.blob
            img = Image.open(BytesIO(image))

            # Perform OCR
            text += "\n" + pytesseract.image_to_string(img)

    return text.strip()

def check_ai_generated(text):
    inputs = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
    outputs = ai_model(**inputs)
    score = torch.softmax(outputs.logits, dim=1)[0][1].item()  # Probability of AI-generated
    return score

def extract_keywords(text, num_keywords=10):
    """Extract meaningful keywords from the OCR-extracted text."""
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())  # Extract words with 4+ letters
    common_words = set(["this", "that", "have", "with", "from", "would", "there", "which"])  # Common words to ignore
    keywords = [word for word in words if word not in common_words]
    
    keyword_counts = Counter(keywords)
    return [word for word, _ in keyword_counts.most_common(num_keywords)]

def google_search_check(text):
    """Perform a Google search to check for plagiarism using extracted keywords."""
    keywords = extract_keywords(text)
    query = " ".join(keywords)  # Create search query using top keywords
    results = []

    try:
        for url in search(query, num_results=3):  # Get top 3 results
            page = requests.get(url, timeout=5)
            soup = BeautifulSoup(page.content, "html.parser")
            page_text = " ".join(p.text for p in soup.find_all("p"))
            results.append((url, page_text))
    except Exception as e:
        print("Google Search Error:", e)
    
    return results

def check_text_similarity(original_text, web_texts):
    texts = [original_text] + [t[1] for t in web_texts]
    vectorizer = TfidfVectorizer().fit_transform(texts)
    vectors = vectorizer.toarray()
    similarities = (vectors @ vectors.T)[0][1:]
    return [(web_texts[i][0], similarities[i]) for i in range(len(web_texts))]

def check_internal_similarity(file_texts):
    """Compare similarity between uploaded files using TF-IDF."""
    file_names = list(file_texts.keys())
    texts = list(file_texts.values())

    vectorizer = TfidfVectorizer().fit_transform(texts)
    vectors = vectorizer.toarray()
    similarity_matrix = vectors @ vectors.T  # Cosine similarity

    internal_similarities = []
    for i in range(len(file_names)):
        for j in range(i + 1, len(file_names)):  # Avoid redundant comparisons
            internal_similarities.append({
                "file1": file_names[i],
                "file2": file_names[j],
                "similarity": similarity_matrix[i][j]
            })

    return sorted(internal_similarities, key=lambda x: x["similarity"], reverse=True)

def generate_pdf(results, internal_similarities):
    pdf_path = os.path.join(REPORT_FOLDER, "plagiarism_report.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    elements = []
    
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    normal_style = styles["Normal"]

    elements.append(Paragraph("Plagiarism Check Results", title_style))
    elements.append(Spacer(1, 12))

    for result in results:
        elements.append(Paragraph(f"File: <b>{result['filename']}</b>", heading_style))
        elements.append(Paragraph(f"<b>AI-Generated Content Score:</b> {round(result['ai_score'], 2)}%", normal_style))
        elements.append(Spacer(1, 6))

        if result["plagiarism_results"]:
            table_data = [["Source URL", "Similarity (%)"]]
            for url, similarity in result["plagiarism_results"]:
                table_data.append([
                    Paragraph(f'<a href="{url}" color="blue">{url}</a>', normal_style), 
                    f"{round(similarity * 100, 2)}%"
                ])
            
            # Define proper column widths to prevent overflow
            table = Table(table_data, colWidths=[400, 80])  
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("WORDWRAP", (0, 0), (-1, -1), True)  # Ensure text wraps properly
            ]))
            elements.append(table)
            elements.append(Spacer(1, 12))

    if internal_similarities:
        elements.append(Paragraph("Similarity Between Uploaded Files", heading_style))
        elements.append(Spacer(1, 6))

        table_data = [["File 1", "File 2", "Similarity (%)"]]
        for sim in internal_similarities:
            file1 = Paragraph(f"<b>{sim['file1']}</b>", normal_style)
            file2 = Paragraph(f"<b>{sim['file2']}</b>", normal_style)
            similarity = f"{round(sim['similarity'] * 100, 2)}%"
            table_data.append([file1, file2, similarity])
        
        # Adjust column widths to ensure names fit properly
        table = Table(table_data, colWidths=[250, 250, 80])  
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("WORDWRAP", (0, 0), (-1, -1), True)  # Wrap long filenames properly
        ]))
        elements.append(table)

    doc.build(elements)
    return pdf_path


@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    internal_similarities = None
    pdf_path = None

    if request.method == "POST":
        files = request.files.getlist("files")  # Get uploaded files
        if not files:
            return "No files uploaded", 400
        
        results = []
        file_texts = {}  # Store file contents for internal similarity check

        for file in files:
            if file.filename.endswith((".pdf", ".docx", ".txt", ".jpg", ".jpeg", ".png")):
                file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(file_path)
                
                text = read_file(file_path)
                file_texts[file.filename] = text  # Store text for internal similarity check
                
                ai_score = check_ai_generated(text)
                web_results = google_search_check(text)
                similarity_scores = check_text_similarity(text, web_results)

                results.append({
                    "filename": file.filename,
                    "ai_score": ai_score,
                    "plagiarism_results": similarity_scores
                })
                
        results.sort(key=lambda x: x["ai_score"], reverse=True)
        internal_similarities = check_internal_similarity(file_texts)
        pdf_path = generate_pdf(results, internal_similarities)

    return render_template("index.html", results=results, internal_similarities=internal_similarities, pdf_path=pdf_path)

@app.route("/download")
def download_pdf():
    pdf_path = os.path.join(REPORT_FOLDER, "plagiarism_report.pdf")
    return send_file(pdf_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)