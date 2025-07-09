import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langchain-groq-chatbot"

prompt= ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions based on the provided context."),   
    ("human", "{input}")])

def generate_response_ChatGroq(question, api_key, llm, temperature, max_tokens):
    llm = ChatGroq(model_name=llm, api_key=api_key, temperature=temperature, max_tokens=max_tokens)
    output_parser = StrOutputParser()
    chain= prompt | llm | output_parser
    response = chain.invoke({"input": question})
    return response

def generate_response_Ollama(question, temperature, max_tokens):
    llm = Ollama(model="gemma:2b")
    output_parser = StrOutputParser()
    chain= prompt | llm | output_parser
    response = chain.invoke({"input": question})
    return response
st.title("LangChain Groq Chatbot")
st.sidebar.header("Configuration")
llm = st.sidebar.selectbox("Select LLM", ["llama-3.3-70b-versatile", "llama-3.1-70b-instruct"])
api_key = st.sidebar.text_input("GROQ API Key", type="password")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 50, 1000, 300)

st.header("Chat with the Bot")
question = st.text_input("Ask a question:")
if question:
    # response=generate_response_ChatGroq(question, api_key, llm, temperature, max_tokens)
    response=generate_response_Ollama(question, temperature, max_tokens)
    st.write("Response:", response)