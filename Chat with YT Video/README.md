# Chat with YouTube Video

This project allows you to chat with any YouTube video. It uses a language model to answer questions based on the video's transcript. You can interact with the application through a Streamlit web interface or a command-line interface.

## Screenshot

![Application Screenshot](Screenshot%202025-07-13%20191528.png)

## Features

- **Chat with any YouTube video:** Provide a YouTube video URL to start a conversation.
- **Web Interface:** A user-friendly web interface built with Streamlit.
- **Command-Line Interface:** A simple command-line version for terminal users.
- **Fast Inference:** Powered by the Groq API for quick responses.
- **RAG Pipeline:** Built with LangChain to retrieve relevant information from the video transcript.
- **HuggingFace Embeddings:** Uses sentence-transformer models for text embeddings.
- **Vector Storage:** ChromaDB is used for efficient vector storage and retrieval.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/chat-with-yt-video.git
   cd chat-with-yt-video
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file:**
   Create a file named `.env` in the root directory and add your API keys:
   ```
   HF_TOKEN="your_huggingface_api_key"
   GROQ_API_KEY="your_groq_api_key"
   ```

## Usage

### Web Application

To run the Streamlit web application:

```bash
streamlit run streamlit_app.py
```

Open your browser and go to the local URL provided by Streamlit.

### Command-Line Interface

To run the command-line application:

```bash
python app.py
```

The application will prompt you to enter a YouTube video URL and then you can start asking questions.

## Dependencies

The project uses the following major libraries:

- `streamlit`
- `python-dotenv`
- `langchain`
- `langchain-core`
- `langchain-community`
- `langchain-groq`
- `langchain-chroma`
- `langchain-huggingface`
- `chromadb`
