# LangChain Chatbot Project

This repository contains a collection of chatbot implementations using LangChain, a framework for developing applications powered by language models.

## Overview

This project demonstrates various types of chatbots using different LLM providers and features:

1. **Simple Chatbot** (`app.py`): A basic Streamlit interface that uses either Groq or Ollama as the LLM provider.
2. **RAG Chatbot** (`chatbot.py`): A conversational chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions based on external web content.
3. **PDF Document RAG Bot** (`ragbot.py`): A Streamlit-based chatbot that answers questions based on content from PDF documents.

## Installation

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai/) (optional, for local LLM usage)

### Setup

1. Clone the repository
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Create a `.env` file in the root directory with the following variables:
```
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
LANGCHAIN_API_KEY=your_langchain_api_key  # Optional for tracing
```

## Usage

### Simple Chatbot (app.py)
```
streamlit run app.py
```
A basic chatbot interface where you can:
- Select different LLM models from Groq
- Configure temperature and max tokens
- Chat with the bot using a simple interface

### RAG Chatbot with Web Content (chatbot.py)
```
python chatbot.py
```
A command-line conversational chatbot that:
- Retrieves information from Wikipedia
- Maintains chat history for contextual responses
- Uses RAG to provide more accurate answers

### PDF Document RAG Bot (ragbot.py)
```
streamlit run ragbot.py
```
A Streamlit interface that:
- Allows uploading and indexing PDF documents
- Creates vector embeddings from documents
- Answers questions based on the document content

## Jupyter Notebook

The repository includes `test.ipynb` which demonstrates core concepts:
- Setting up LLM models
- Conversation history management
- Vector stores and retrievers

## Project Structure

```
├── app.py               # Simple chatbot with Streamlit UI
├── chatbot.py           # RAG-based chatbot with web content
├── ragbot.py            # PDF document chatbot with Streamlit UI
├── requirements.txt     # Project dependencies
├── test.ipynb           # Jupyter notebook with examples
└── pdf_docs/            # Directory for PDF documents
```

## Technologies Used

- **LangChain**: Framework for LLM applications
- **Groq**: Fast LLM inference API
- **Ollama**: Local LLM deployment
- **Hugging Face**: Embeddings and models
- **FAISS/Chroma**: Vector databases for document retrieval
- **Streamlit**: Web interface
- **PyPDF**: PDF document parsing

