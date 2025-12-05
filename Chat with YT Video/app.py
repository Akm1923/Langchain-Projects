from langchain_community.document_loaders import YoutubeLoader
"""
This script loads a YouTube video transcript, splits it into manageable text chunks, embeds the chunks using a HuggingFace model, and stores them in a Chroma vector database. It then sets up a retrieval-augmented generation (RAG) pipeline using a Groq LLM to answer user questions based on the retrieved context from the video transcript.
Workflow:
1. Loads environment variables for API keys.
2. Downloads and processes a YouTube video's transcript.
3. Splits the transcript into overlapping text chunks.
4. Embeds the chunks and stores them in a persistent Chroma vector store.
5. Sets up a retriever to fetch relevant chunks using Maximal Marginal Relevance (MMR).
6. Defines a prompt template for question answering.
7. Initializes a Groq LLM for generating answers.
8. Constructs a chain that retrieves context, formats the prompt, queries the LLM, and parses the output.
9. Invokes the chain with a sample question and prints the answer.
Dependencies:
- langchain_community
- langchain_core
- langchain.text_splitter
- langchain_huggingface
- langchain_chroma
- langchain.prompts
- langchain_groq
- python-dotenv
- os
Environment Variables:
- HF_TOKEN: HuggingFace API token
- GROQ_API_KEY: Groq API key
Example usage:
    python app.py
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel,RunnableSequence,RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers.string import StrOutputParser

import os
from dotenv import load_dotenv
load_dotenv()
hf_api_key=os.getenv("HF_TOKEN")
groq_api_key=os.getenv("GROQ_API_KEY")

yt_link=str(input("Enter the link of the video from which you want to chat:"))
#loading the transcript of youtube video
loader = YoutubeLoader.from_youtube_url(
    yt_link, add_video_info=False
)
doc=loader.load()

#splitting the documnet into chunks
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200)

docs=splitter.split_documents(doc)

#huggingface embeddings inference
model = "sentence-transformers/all-MiniLM-L6-v2"
hf = HuggingFaceEndpointEmbeddings(
    model=model,
    huggingfacehub_api_token=hf_api_key
)


#creating vector store
vector_store = Chroma(
    embedding_function=hf,
    persist_directory="chroma db" # Where to save data locally, remove if not necessary
)
vector_store.add_documents(docs)

#creating a retiver
retriver=vector_store.as_retriever(search_type="mmr",search_kwargs={"k":3},lambda_mult=0.6)


#prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Answer the question based on the context below if no context available simply say no context is given about this in the video.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
)

#llm
model= ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.1,
    max_tokens=512,
    api_key=groq_api_key
)

#string output parser
parser = StrOutputParser()

#parallel chain
parallel = RunnableParallel(
    question=RunnablePassthrough(),
    context=retriver)

#main chain
chain = parallel | prompt | model | parser

#final output
while True:
    question = str(input("Enter the question you want to ask from the video:"))
    if question.lower() == "exit":
        break
    print(chain.invoke(question))