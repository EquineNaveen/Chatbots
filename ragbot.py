import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm=ChatGroq(
    model_name="llama-3.3-70b-versatile")

prompt=ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions based on the provided context."
     "{context}"),
    ("user", "{input}")
])

def create_vector_embeddings():
    embeddings= OllamaEmbeddings(model="gemma:2b")
    loader=PyPDFDirectoryLoader("pdf_docs")
    docs=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    docs=text_splitter.split_documents(docs)
    vectorstore=FAISS.from_documents(docs,embedding=embeddings) 
    return vectorstore

# Initialize session state to store vectorstore
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

st.sidebar.header("Configuration")
if st.sidebar.button("Create Vector Embeddings"):
    st.session_state.vectorstore = create_vector_embeddings()
    st.sidebar.success("Vector embeddings created successfully!")

inputs=st.text_input("Enter Your query")
if st.button("Submit"):
    # Check if vectorstore exists before using it
    if st.session_state.vectorstore is None:
        st.error("Please create vector embeddings first by clicking the button in the sidebar.")
    else:
        vectors = st.session_state.vectorstore.as_retriever(search_kwargs={"k":1})
        # Fix the history_aware_retriever creation - it needs to be called properly
        history_aware_retriever = create_history_aware_retriever(llm, vectors, prompt)
        question_answer_chain = create_stuff_documents_chain(llm, prompt=prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

      
        # Fix the syntax error in using the input variable
        response =rag_chain.invoke({"input": inputs})
        st.write("Response:", response["answer"])