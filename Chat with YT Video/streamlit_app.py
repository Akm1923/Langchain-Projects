import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
import os

# Load API keys
load_dotenv()
hf_api_key = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

# Set Streamlit page config
st.set_page_config(page_title="YouTube Video Chatbot", layout="wide")

st.markdown("""
    <style>
        .stTextInput, .stTextArea {border-radius: 10px;}
        .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px;}
    </style>
""", unsafe_allow_html=True)

st.title("🎥 YouTube Video Chatbot")
st.markdown("Ask questions based on the **video transcript** using **Groq + LangChain**.")

# Sidebar for input
with st.sidebar:
    yt_link = st.text_input("🔗 Enter YouTube Video URL:", "")
    if yt_link:
        st.video(yt_link,width="stretch")

if yt_link:
    with st.spinner("🔄 Processing video transcript... please wait (few seconds)"):
        # Load transcript
        loader = YoutubeLoader.from_youtube_url(yt_link, add_video_info=False)
        doc = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(doc)

        # Embedding
        embed_model = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=hf_api_key
        )

        # Vector DB
        vector_store = Chroma(
            embedding_function=embed_model,
            persist_directory="chroma_db"
        )
        vector_store.add_documents(docs)

        # Retriever
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3}, lambda_mult=0.6)

        # Prompt Template
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "Answer the question based on the context below. "
                "If there's no relevant context, say: 'No relevant context from video transcript.'\n\n"
                "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            )
        )

        # LLM
        llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.1,
            max_tokens=512,
            api_key=groq_api_key
        )

        # Chain
        parser = StrOutputParser()
        chain = (
            RunnableParallel(question=RunnablePassthrough(), context=retriever)
            | prompt
            | llm
            | parser
        )

    # Chat Interface
    st.subheader("💬 Ask a question about the video:")
    user_query = st.text_input("Type your question here...", "")
   

    if  st.button("Search"):
        with st.spinner("🤖 Thinking..."):
            answer = chain.invoke(user_query)
            st.markdown(f"**Answer:** {answer}")
