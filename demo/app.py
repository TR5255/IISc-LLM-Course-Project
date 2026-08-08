import os
import sys
import tempfile

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import PyPDF2


# ============================================================
# Project paths
# ============================================================

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(DEMO_DIR)

LLM_DIR = os.path.join(PROJECT_DIR, "llm")

if LLM_DIR not in sys.path:
    sys.path.insert(0, LLM_DIR)


# Import the RAG pipeline
from rag_pipeline import LegalRAG


# ============================================================
# Flask
# ============================================================

app = Flask(
    __name__,
    template_folder=os.path.join(
        DEMO_DIR,
        "templates"
    ),
    static_folder=os.path.join(
        DEMO_DIR,
        "static"
    ),
)

app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


# ============================================================
# Load RAG model once
# ============================================================

print("=" * 60)
print("Loading Legal RAG system...")
print("=" * 60)

rag = LegalRAG()

print("=" * 60)
print("Legal RAG system ready")
print("=" * 60)


# ============================================================
# PDF extraction
# ============================================================

def extract_pdf_text(file):

    reader = PyPDF2.PdfReader(file)

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


# ============================================================
# Convert document into passages
# ============================================================

def create_passages(text):

    # Basic paragraph-based splitting.
    # We will improve this later if necessary.

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    # If PDF extraction doesn't preserve paragraphs,
    # fall back to line-based chunks.
    if len(paragraphs) <= 1:

        paragraphs = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

    return paragraphs


# ============================================================
# Routes
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        # ----------------------------------------------------
        # Validate upload
        # ----------------------------------------------------

        if "contract" not in request.files:

            return jsonify({
                "error": "Please upload a contract."
            }), 400

        file = request.files["contract"]

        if file.filename == "":

            return jsonify({
                "error": "No file selected."
            }), 400

        if not file.filename.lower().endswith(".pdf"):

            return jsonify({
                "error": "Only PDF files are supported."
            }), 400

        # ----------------------------------------------------
        # Question
        # ----------------------------------------------------

        question = request.form.get(
            "question",
            ""
        ).strip()

        if not question:

            return jsonify({
                "error": "Please enter a question."
            }), 400

        # ----------------------------------------------------
        # Extract PDF
        # ----------------------------------------------------

        text = extract_pdf_text(file)

        if not text.strip():

            return jsonify({
                "error": (
                    "Could not extract text from this PDF. "
                    "The document may be scanned or image-based."
                )
            }), 400

        # ----------------------------------------------------
        # Create passages
        # ----------------------------------------------------

        passages = create_passages(text)

        if not passages:

            return jsonify({
                "error": "No usable text was found in the contract."
            }), 400

        print()
        print("=" * 60)
        print("NEW REQUEST")
        print("=" * 60)

        print(
            f"Contract: {secure_filename(file.filename)}"
        )

        print(
            f"Passages: {len(passages)}"
        )

        print(
            f"Question: {question}"
        )

        # ----------------------------------------------------
        # Run RAG
        # ----------------------------------------------------

        result = rag.answer(
            question,
            passages
        )

        # ----------------------------------------------------
        # Prepare response
        # ----------------------------------------------------

        reranked = []

        for rank, item in enumerate(
            result["reranked"],
            1
        ):

            reranked.append({
                "rank": rank,
                "text": item["text"],
                "retriever_score": item[
                    "retriever_score"
                ],
                "reranker_score": item[
                    "reranker_score"
                ],
            })

        return jsonify({
            "success": True,
            "answer": result["answer"],
            "passages": reranked,
        })

    except Exception as e:

        print()
        print("ERROR:")
        print(e)

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),
        debug=True,
    )