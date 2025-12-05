# Langchain Projects

This repository contains a collection of projects built using Langchain and related technologies. Each project is self-contained in its own directory.

## Projects

Here is a list of the projects in this repository:

### Chat with YT Video
- **Directory:** `Chat with YT Video/`
- **Description:** A Streamlit application that allows you to chat with a YouTube video. You provide a YouTube video URL, and the application will answer your questions about the video's content.
- **Key Technologies:** Langchain, Streamlit, YouTube Transcript API.

### End to End Conversational QNA Chatbot
- **Directory:** `End to End Conversational QNA Chatbot/`
- **Description:** A conversational Q&A chatbot that can answer questions from a PDF document. This project demonstrates how to create a chatbot that can understand context and provide relevant answers from a knowledge base.
- **Key Technologies:** Langchain, ChromaDB, Hugging Face Transformers.

### End To End QNA chatbot
- **Directory:** `End To End QNA chatbot/`
- **Description:** A simple Q&A chatbot that answers questions based on a given context.
- **Key Technologies:** Langchain, FAISS.

### RAG Document Q&A with Groq API
- **Directory:** `RAG Document Q&A with Groq API/`
- **Description:** A document question-and-answer application that uses the Groq API for fast inference. This project showcases the use of Retrieval-Augmented Generation (RAG) with a high-performance inference engine.
- **Key Technologies:** Langchain, Groq API, FAISS.

## Getting Started

To run any of the projects, navigate to the project's directory and install the required dependencies from the `requirements.txt` file:

```bash
cd <project-directory>
pip install -r requirements.txt
```

Then, run the main application file (usually `app.py` or `streamlit_app.py`):

```bash
python app.py
```
or
```bash
streamlit run streamlit_app.py
```
