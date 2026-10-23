
# RAG PDF Chat

A mobile-friendly PDF question-answering web app built
with Python and Streamlit.

## Features
- Upload text-based PDF documents
- Extract text page by page
- Split text into overlapping chunks
- Retrieve relevant passages using TF-IDF
- Optional LLM-generated answers using Groq
- Show retrieved passages and source page numbers
- Fallback to local retrieval if the API is unavailable

## Tech Stack
- Python
- Streamlit
- pypdf
- scikit-learn
- NumPy
- Requests
- Optional Groq API

## Run locally

Install dependencies:

    pip install -r requirements.txt

Start the app:

    streamlit run app.py

## Limitations
- Scanned PDFs require OCR.
- TF-IDF is lexical retrieval, not neural semantic search.
- LLM responses require an available API key and quota.
- Retrieved passages can be irrelevant; verify important answers.
- 
