# Simple Q&A Chatbot using Groq

This project is a simple chatbot application built with Streamlit, LangChain, and Groq. It allows users to ask questions and get responses from various large language models (LLMs) available through the Groq API.

## Screenshot

![Chatbot Screenshot](Screenshot%202025-07-05%20015615.png)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repository-name.git
   cd your-repository-name
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create a `.env` file:**
   Create a `.env` file in the root directory of the project and add your Groq API key and Langsmith details:
   ```env
   GROQ_API_KEY="your-groq-api-key"
   LANGSMITH_API_KEY="your-langsmith-api-key"
   LANGSMITH_PROJECT="your-langsmith-project-name"
   LANGSMITH_TRACING="true"
   ```

## Usage

1. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser:**
   The app will open in your default web browser, usually at `http://localhost:8501`.

3. **Interact with the chatbot:**
   - Enter your Groq API key in the sidebar.
   - Select the desired LLM from the dropdown menu.
   - Adjust the temperature and max tokens using the sliders.
   - Type your question in the text input box and click "Generate Response".

## Built With

- [Streamlit](https://streamlit.io/) - The web framework used.
- [LangChain](https://www.langchain.com/) - The framework for developing applications powered by language models.
- [Groq](https://groq.com/) - The API for accessing large language models.
- [Python](https://www.python.org/) - The programming language used.

## Acknowledgments

Made with ❤️ by [AKM]
