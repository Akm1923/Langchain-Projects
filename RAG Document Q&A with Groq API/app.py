import os
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain


# Load environment variables

load_dotenv()

# LangSmith tracing configuration (optional)
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")

# API keys
groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")
os.environ['HF_TOKEN'] = hf_token


# Initialize LLM from Groq

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=groq_api_key
)


# Streamlit App Interface

st.title("📄 RAG Document Q&A with Groq")
st.text("Ask a question based on the paper: 'Attention Is All You Need'\n(Located in the 'research_papers' folder)")


# Prompt Template for Q&A

template = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant. Answer the question based on the context provided.
    <context>
    {context}
    <context>
    question: {input}
    """
)


# Function: Create Vector Embeddings

def create_vector_embeddings():
    """
    Initializes and stores vector embeddings for PDF documents in the Streamlit session state.
    This function performs the following steps:
    1. Checks if vector embeddings have already been created in the session state.
    2. Initializes the embedding model (OllamaEmbeddings with "llama3.2").
    3. Loads PDF documents from the "research_papers" directory.
    4. Splits the loaded documents into manageable text chunks.
    5. Stores the loader, documents, text splitter, and chunks in the session state.
    6. Creates a FAISS vector store from the text chunks and embeddings, and stores it in the session state.
    Note:
        This function is intended to be run only once per session to avoid redundant computation.
    """
    # Only run once per session
    if "vectors" not in st.session_state:
        # st.session_state.embeddings = HuggingFaceEmbeddings(
        #     model_name="Qwen/Qwen3-Embedding-0.6B"
        # )
        st.session_state.embeddings = OllamaEmbeddings(model="llama3.2")

        # Load PDF documents from local folder
        loader = PyPDFDirectoryLoader("research_papers")
        documents = loader.load()

        # Split long documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        docs = text_splitter.split_documents(documents)

        # Store in session
        st.session_state.loader = loader
        st.session_state.documents = documents
        st.session_state.text_splitter = text_splitter
        st.session_state.docs = docs

        # Create a FAISS vector store from the chunks
        st.session_state.vectors = FAISS.from_documents(
            docs,
            st.session_state.embeddings
        )


# Embedding Button

if st.button("🔍 Generate Document Embedding"):
    create_vector_embeddings()
    st.success("✅ Document embeddings created successfully!")


# Question Input

user_prompt = st.text_input("💬 Enter your question:")


# Handle Q&A Response

if st.button("🧠 Get Answer") and user_prompt:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please generate document embeddings first.")
    else:
        # Create the retrieval and document chain
        retriever = st.session_state.vectors.as_retriever()
        document_chain = create_stuff_documents_chain(llm=llm, prompt=template)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Get response
        response = retrieval_chain.invoke({"input": user_prompt})

        # Show final answer
        st.markdown("### 🧠 Answer:")
        st.write(response["answer"])

        # Show document context
        with st.expander("📚 See Retrieved Context"):
            for doc in response["context"]:
                st.write(doc.page_content)
                st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")