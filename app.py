from flask import Flask, render_template, request, send_file
import os
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from googlesearch import search
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from pdfminer.high_level import extract_text
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

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
    ext = file_path.split(".")[-1]
    if ext == "pdf":
        return extract_text(file_path)
    elif ext == "docx":
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def check_ai_generated(text):
    inputs = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
    outputs = ai_model(**inputs)
    score = torch.softmax(outputs.logits, dim=1)[0][1].item()  # Probability of AI-generated
    return score

def google_search_check(text):
    query = " ".join(text.split()[:10])  # Take first 10 words for search
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

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        files = request.files.getlist("files")  # Get uploaded files
        if not files:
            return "No files uploaded", 400
        
        results = []
        file_texts = {}  # Store file contents for internal similarity check

        for file in files:
            if file.filename.endswith((".pdf", ".docx", ".txt")):
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

        internal_similarities = check_internal_similarity(file_texts)
        pdf_path = generate_pdf(results, internal_similarities)
        return render_template("result.html", results=results, internal_similarities=internal_similarities, pdf_path=pdf_path)

    return render_template("index.html")

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
        elements.append(Paragraph(f"File: {result['filename']}", heading_style))
        elements.append(Paragraph(f"<b>AI-Generated Content Score:</b> {round(result['ai_score'], 2)}", normal_style))
        elements.append(Spacer(1, 6))

        if result["plagiarism_results"]:
            table_data = [["Source URL", "Similarity (%)"]]
            for url, similarity in result["plagiarism_results"]:
                table_data.append([Paragraph(f'<a href="{url}" color="blue">{url}</a>', normal_style), f"{round(similarity * 100, 2)}%"])
            
            table = Table(table_data, colWidths=[350, 100])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")
            ]))
            elements.append(table)
            elements.append(Spacer(1, 12))

    doc.build(elements)
    return pdf_path

@app.route("/download")
def download_pdf():
    pdf_path = os.path.join(REPORT_FOLDER, "plagiarism_report.pdf")
    return send_file(pdf_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
