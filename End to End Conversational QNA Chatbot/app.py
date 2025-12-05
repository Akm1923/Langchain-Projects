import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# Load API keys
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

if not hf_token or not groq_api_key:
    st.error("❌ Please set HF_TOKEN and GROQ_API_KEY in your .env file.")
    st.stop()

# Page title and sidebar
st.set_page_config(page_title="Conversational PDF Chatbot", layout="wide")
st.title("📄🔍 End-to-End Conversational RAG Chatbot")
st.sidebar.header("📎 Upload PDF File(s):")
uploaded_files = st.sidebar.file_uploader("Choose PDF file(s)", accept_multiple_files=True, type="pdf")

# Initialize session state
if "store" not in st.session_state:
    st.session_state.store = {}
if "chat_input_key" not in st.session_state:
    st.session_state.chat_input_key = 0

# Main logic
if uploaded_files:
    documents = []
    temp_files = []

    try:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_files.append(tmp_file.name)
                loader = PyPDFLoader(tmp_file.name)
                docs = loader.load()
                documents.extend(docs)

        st.sidebar.success("✅ All PDF documents loaded successfully!")

        # Chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(documents)

        # Embeddings
        embeddings = HuggingFaceEndpointEmbeddings(
            huggingfacehub_api_token=hf_token,
            model="BAAI/bge-large-en-v1.5"
        )

        # Vector store
        persist_dir = "./chroma_langchain_db"
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)  # Clear previous vector store
        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        retriever = vector_store.as_retriever()

        # LLM
        llm = ChatGroq(
            model="llama3-8b-8192",
            temperature=1.2,
            api_key=groq_api_key
        )

        # Setup retriever prompt
        retriever_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Reformulate the user question by considering the chat history to improve document retrieval."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            prompt=retriever_prompt,
            retriever=retriever
        )

        # Setup document answer prompt
        doc_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer the user's question based on the following context from retrieved documents:\n\n{context}\n\nProvide a detailed response with a minimum of 50 words. If the context does not contain relevant information, state so and provide a general response to the best of your ability."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        document_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=doc_prompt
        )

        rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

        # Setup chat history storage
        def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        chain_with_history = RunnableWithMessageHistory(
            rag_chain,
            get_by_session_id,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Chat Interface
        st.subheader("💬 Ask any question about your uploaded PDF(s):")
        user_input = st.text_input("Your Question:", key=f"chat_input_{st.session_state.chat_input_key}")

        if user_input:
            with st.spinner("🔎 Thinking..."):
                try:
                    response = chain_with_history.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": "default_session_1"}}
                    )
                    st.markdown("### 🧠 Answer:")
                    st.write(response["answer"])
                    # Increment key to clear input field
                    st.session_state.chat_input_key += 1
                except Exception as e:
                    st.error(f"❌ Error processing question: {str(e)}")

    except Exception as e:
        st.error(f"❌ Error processing PDF files: {str(e)}")
    
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

else:
    st.info("📤 Please upload at least one PDF file to begin.")