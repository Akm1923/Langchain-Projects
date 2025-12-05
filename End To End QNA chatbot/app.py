import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

#langchain tracking
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["LANGSMITH_TRACING"]=os.getenv("LANGSMITH_TRACING")


#prompt template
prompt=ChatPromptTemplate.from_messages(
     [
          ("system", "You are a helpful assistant that answers questions that user asks"),
          ("human", "question: {question}")
     ]
)


def generate_response(question, api_key,llm,temperature,max_tokens):
     groq_api_key=api_key
     model=ChatGroq(model=llm,api_key=groq_api_key,temperature=temperature,max_tokens=max_tokens)
     parser=StrOutputParser()
     chain=prompt|model|parser
     result=chain.invoke({"question":question})
     return result


#setting the title of the app
st.title("Simple Q&A Chatbot using Groq")


#getting the api key
api_key=st.sidebar.text_input("Enter the API Key", type="password")

#getting the llm model
llm=st.sidebar.selectbox("Select the LLM", ["gemma2-9b-it","distil-whisper-large-v3-en", "llama-3.1-8b-instant","meta-llama/llama-guard-4-12b"])

#setting the temperature
temperature=st.sidebar.slider("Select the Temperature", 0.0, 1.0, 0.2)

#setting the max tokens
max_tokens=st.sidebar.slider("Select the Max Tokens", 1, 1000, 50)

st.write("Ask a question to the chatbot")
#taking the user input
question=st.text_input("Enter your question")
#button to generate the response
if st.button("Generate Response"):
     if question and api_key:
          with st.spinner("Generating response..."):
               response=generate_response(question, api_key,llm,temperature,max_tokens)
          st.write("Response: ", response)
     else:
          st.error("Please enter a question and API key")
#footer
st.markdown("---")
st.markdown("Made with ❤️ by [AKM]")


     

