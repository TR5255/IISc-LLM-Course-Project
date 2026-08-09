
---

# Legal-Salience

## Legal Contract Question Answering using Retrieval-Augmented Generation

Legal-Salience is a domain-specific Retrieval-Augmented Generation (RAG) system for answering questions about legal contracts.

The system combines:

* A domain-adapted BGE dense retriever
* A fine-tuned BGE reranker
* Gemini for answer generation
* Contract-level and sentence-level retrieval
* Evidence-grounded answer generation
* A Flask-based web application for interactive contract analysis

The goal is to provide concise answers to contract-related questions while also showing the relevant contractual language used as supporting evidence.

---

## System Overview

The system follows a multi-stage retrieval and generation pipeline:

Contract PDF
↓
Text Extraction
↓
Sentence / Passage Processing
↓
BGE Legal Retriever
↓
Top-K Passages
↓
Fine-tuned BGE Reranker
↓
Top-N Relevant Passages
↓
Gemini LLM
↓
Answer Generation
↓
Answer + Supporting Clauses

The retrieval stage identifies potentially relevant contractual passages, while the reranker improves their ordering before the most relevant passages are provided to the LLM.

---

## Key Features

### 1. Legal Passage Retrieval

The system uses a BGE-based dense retriever trained/adapted for legal contract retrieval.

Given a user question, the retriever searches the contract passages and returns the most semantically relevant candidates.

### 2. Passage Reranking

Retrieved candidates are passed through a fine-tuned BGE reranker.

This provides a second-stage ranking mechanism that improves the ordering of relevant contractual provisions.

### 3. LLM-Based Answer Generation

The highest-ranked contractual passages are provided to Gemini together with the user's question.

The LLM generates an answer based on the retrieved contract evidence rather than relying only on its general knowledge.

### 4. Evidence Display

The web application displays the relevant contract excerpts used to support the generated answer.

This makes the system more transparent and allows users to inspect the underlying contractual language.

### 5. Interactive Web Application

A Flask application provides a simple interface where users can:

1. Upload a PDF contract
2. Enter a question
3. Run the Legal RAG pipeline
4. View the generated answer
5. View the retrieved supporting clauses

---

# Dataset

The project uses the CUAD (Contract Understanding Atticus Dataset) for contract question-answering and retrieval evaluation.

The dataset contains contracts together with questions and annotations identifying relevant contractual provisions.

The repository contains processed contract indexes used by the retrieval and evaluation pipeline.

Important files include:

data/

* contract_index.json
* CUAD.json
* test_contract_index.json
* train_contract_index.json
* train_bge_hybrid.json
* train_bge_hybrid.jsonl
* train_bge_hybrid.py
* train_contract_index.json
* train_reranker_bge_same_contract_hn.jsonl
* train_seperate_questions.json

The processed contract indexes represent contracts as collections of textual passages/sentences, allowing the retriever and reranker to operate directly over individual contract passages.

---

# Repository Structure

Legal-Salience/
├── data/
│   ├── contract_index.json
│   ├── CUAD.json
│   ├── test_contract_index.json
│   ├── train_contract_index.json
│   ├── train_bge_hybrid.json
│   ├── train_bge_hybrid.jsonl
│   ├── train_bge_hybrid.py
│   ├── train_reranker_bge_same_contract_hn.jsonl
│   └── train_seperate_questions.json
│
├── datasets/
│   ├── build_contract_index.py
│   ├── load_cuad.py
│   └── prepare_eval_data.py
│
├── demo/
│   ├── templates/
│   │   └── index.html
│   ├── static/
│   │   ├── style.css
│   │   └── script.js
│   ├── uploads/
│   ├── app.py
│   └── requirements.txt
│
├── llm/
│   ├── evaluate_llm_metrics.py
│   ├── evaluate_llm.py
│   ├── prompt.py
│   ├── rag_pipeline.py
│   └── test_llm.py
│
├── models/
│   ├── convert_tobge_jsonl.py
│   ├── create_reranker_train.py
│   ├── evaluate_flagembedding.py
│   └── evaluate_reranker.py
│
├── notebooks/
│   ├── 01_Finetuned_Retriever.ipynb
│   ├── 02_Finetuned_Reranker.ipynb
│   └── 03_LLM_Integration_And_Evaluation.ipynb
│
├── outputs/
│   └── llm_evaluation_results.jsonl
│
├── .gitignore
├── README.md
└── requirements.txt

Some generated/intermediate files may not be included in every checkout.

---

# Installation

## 1. Clone the repository

```
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Legal-Salience
```

## 2. Create a virtual environment

### Windows

```
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```
python -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```
pip install -r requirements.txt
```

For the web application, install the additional dependencies if required:

```
pip install -r demo/requirements.txt
```

---

# Gemini API Configuration

The LLM generation stage uses the Gemini API.

The API key should not be hard-coded into the source code.

### Windows PowerShell

```
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

Verify that the variable is available:

```
echo $env:GEMINI_API_KEY
```

### Linux / macOS

```
export GEMINI_API_KEY="YOUR_API_KEY"
```

The API key should never be committed to GitHub.

Make sure the following is included in `.gitignore` if you use local environment files:

```
.env
*.key
```

---

# Running the RAG Pipeline

The main RAG implementation is located at:

```
llm/rag_pipeline.py
```

The main pipeline performs:

Question
↓
Dense Retrieval
↓
Candidate Passages
↓
Reranking
↓
Top Relevant Passages
↓
Gemini
↓
Answer

The pipeline can also return the retrieved and reranked passages used to generate the answer.

---

# Running the Test Evaluation

The LLM evaluation script evaluates the complete RAG pipeline on the CUAD test questions.

Run:

```
python llm/test_llm.py
```

The generated results can be saved as JSONL for later analysis.

Example output:

```
Questions tested : 1244
Saved to         : llm_evaluation_results.jsonl
```

---

# Quantitative Evaluation

The system was evaluated on 1,244 CUAD test questions.

The current evaluation uses a compact set of retrieval, semantic-answer, and system reliability metrics.

## Retrieval / Reranking

| Metric   |  Score |
| -------- | -----: |
| Recall@1 | 66.93% |
| Recall@3 | 80.21% |
| MRR      | 0.7313 |

### Recall@1

Recall@1 measures whether at least one annotated relevant contractual passage appears at the first position of the reranked results.

### Recall@3

Recall@3 measures whether a relevant contractual passage appears within the top three reranked passages.

### Mean Reciprocal Rank

MRR measures how highly the first relevant passage is ranked. Higher values indicate that relevant contractual evidence tends to appear closer to the top of the ranking.

---

## LLM Evaluation

| Metric                           |  Score |
| -------------------------------- | -----: |
| Evidence Semantic Similarity     | 0.8613 |
| Answer Relevance                 | 0.8249 |
| Ground-truth Semantic Similarity | 0.8461 |
| Generation Success Rate          | 99.84% |

The semantic metrics are embedding-based similarity measures and should not be interpreted as human-annotated accuracy percentages.

The generation success rate indicates the proportion of evaluation questions for which the complete LLM generation pipeline successfully returned an answer.

---

# LLM Evaluation Script

The quantitative evaluation can be reproduced using:

```
python llm/evaluate_llm_metrics.py
```

The script evaluates:

* Recall@1
* Recall@3
* MRR
* Evidence semantic similarity
* Answer relevance
* Ground-truth semantic similarity
* Generation success rate

The evaluation uses the previously generated LLM responses and therefore does not require regenerating all answers.

---

# Web Application

The project includes a Flask-based demonstration application in:

```
demo/
```

The application provides:

* PDF contract upload
* Natural-language question input
* RAG-based analysis
* Generated answer display
* Supporting contract excerpts
* Reranker scores for retrieved passages

## Run the application

From the project root:

```
cd demo
python app.py
```

The application will start a local Flask server.

Open the displayed local address in a browser.

---

# Example Questions

The system can answer questions such as:

```
Which state's law governs the interpretation of this agreement?

What notice period is required to terminate the agreement?

Does the agreement automatically renew?

What are the termination conditions?

What happens if either party breaches the agreement?

What is the governing law and jurisdiction?

What are the obligations of the parties?
```

The application returns both the generated response and the contractual excerpts used as evidence.

---

# Model Components

The project contains separate components for retrieval, reranking, and generation.

### Retriever

A BGE-based legal retriever is used for dense semantic retrieval.

### Reranker

A fine-tuned BGE reranker performs second-stage relevance ranking over the retrieved passages.

### LLM

Gemini is used as the final answer-generation model.

The LLM receives the question together with the highest-ranked contractual evidence.

---

# Notebooks

The `notebooks/` directory contains experiments and development notebooks related to:

```
01_Finetuned_Retriever.ipynb
02_Finetuned_Reranker.ipynb
03_LLM_Integration_And_Evaluation.ipynb
```

These notebooks document model development, training, and integration experiments.

---

# Reproducibility

The project separates the main stages of the system:

Dataset
↓
Preprocessing
↓
Retriever
↓
Reranker
↓
LLM
↓
Evaluation
↓
Web Application

Processed datasets, training scripts, evaluation scripts, and model-related notebooks are maintained separately to make the individual stages easier to reproduce.

---

# Limitations

The system has several limitations:

* Retrieval performance depends on the quality of the contract text extraction and passage segmentation.
* A relevant passage may not always appear in the top-ranked results.
* LLM responses depend on the quality and completeness of retrieved evidence.
* Embedding-based semantic similarity metrics are proxies and are not equivalent to human legal-accuracy judgments.
* The system is intended as an AI-assisted contract analysis tool and should not replace review by a qualified legal professional.

---

# Future Work

Potential improvements include:

* Better contract-specific passage segmentation
* Improved retrieval and reranking models
* Multi-passage reasoning
* More robust citation and evidence attribution
* Human evaluation of answer correctness and faithfulness
* Evaluation across additional legal contract datasets
* Improved handling of long contracts and cross-clause dependencies

---

