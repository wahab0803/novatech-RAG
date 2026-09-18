# NovaTech RAG

An end-to-end Retrieval-Augmented Generation (RAG) system built as a practical learning project.

The system allows users to ask questions about a fictional company called NovaTech Solutions. Instead of relying only on the language model's internal knowledge, the system retrieves relevant information from a local company knowledge base and provides that context to Gemini before generating an answer.

---

## Overview

This project demonstrates the core architecture of a RAG pipeline:

User Question
        ↓
Query Embedding
        ↓
ChromaDB Retrieval
        ↓
Relevant Knowledge
        ↓
Gemini
        ↓
Generated Answer

The goal of the project was to understand how retrieval and generation work together rather than simply calling an LLM directly.

---

## Features

- Load knowledge from multiple `.txt` documents
- Split documents into smaller chunks
- Generate vector embeddings using Sentence Transformers
- Store embeddings in ChromaDB
- Retrieve relevant information using semantic similarity
- Pass retrieved context to Gemini
- Generate grounded answers
- Prevent the model from using outside knowledge when the answer is not present in the knowledge base

---

## Technologies & Tools

### Programming Language

- Python

### AI / LLM

- Google Gemini
- Gemini 2.5 Flash-Lite

### Embeddings

- Sentence Transformers
- `all-MiniLM-L6-v2`

### Vector Database

- ChromaDB

### Environment Management

- Python Virtual Environment
- python-dotenv

### Development Tools

- PyCharm / IntelliJ-based IDE
- PowerShell
- Git
- GitHub

---

## Project Structure

```text
novatech-rag/
│
├── app/
│   ├── main.py
│   └── test_gemini.py
│
├── documents/
│   ├── company_profile.txt
│   ├── contact_information.txt
│   ├── faqs.txt
│   ├── pricing.txt
│   ├── refund_policy.txt
│   └── services.txt
│
├── .gitignore
├── README.md
└── requirements.txt
