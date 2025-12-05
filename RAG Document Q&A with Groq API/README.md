# RAG Document Q&A with Groq API

This project implements a Retrieval-Augmented Generation (RAG) system to answer questions about documents stored locally. It uses the Groq API for high-speed language model inference, providing fast and accurate answers from your own knowledge base, with a simple web interface built with Streamlit.

## Screenshots

![Screenshot 1](Screenshot%202025-07-13%20184827.png)
![Screenshot 2](Screenshot%202025-07-13%20184840.png)

## Features

- **Chat with Your Documents**: Ask questions in natural language about the content of your PDF files.
- **High-Speed Inference**: Powered by the Groq API (`llama-3.1-8b-instant`) for near-instant responses.
- **Local Knowledge Base**: Uses PDF documents in the `research_papers` directory as the source of truth.
- **Simple UI**: Easy-to-use web interface built with Streamlit.
- **Local Embeddings**: Uses Ollama (`llama3.2`) to generate document embeddings locally.

## Project Structure

```
.
├── .env
├── app.py
├── requirements.txt
└── research_papers/
    └── 1706.03762v7.pdf
```

- **`app.py`**: The main Streamlit application script.
- **`requirements.txt`**: A list of all the Python packages required to run the project.
- **`.env`**: Configuration file for storing environment variables like API keys.
- **`research_papers/`**: Directory where you should place your PDF documents to be used as the knowledge base.

## Setup and Installation

Follow these steps to set up and run the project locally.

### 1. Prerequisites

- Python 3.8+
- An API key from [Groq](https://console.groq.com/keys).
- [Ollama](https://ollama.com/) installed and running locally with the `llama3.2` model (`ollama pull llama3.2`).

### 2. Clone the Repository

```bash
git clone <repository-url>
cd RAG-Document-Q-A-with-Groq-API
```

### 3. Create a Virtual Environment

It is recommended to use a virtual environment to manage project dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 4. Install Dependencies

Install all the required packages from `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a file named `.env` in the root of the project directory and add your Groq API key.

```env
GROQ_API_KEY="gsk_your_api_key_here"
```

### 6. Add Documents

Place any PDF files you want to query inside the `research_papers` directory.

## Usage

1.  Ensure your Ollama server is running.
2.  Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

3.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).
4.  Click the **"Generate Document Embedding"** button to process and index the documents.
5.  Once the embeddings are created, type your question into the text input and click **"Get Answer"**.

