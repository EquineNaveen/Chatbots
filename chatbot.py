import os
import bs4
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain.chains import create_history_aware_retriever
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")

llm=ChatGroq(model_name="llama-3.3-70b-versatile",api_key=groq_api_key)
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

loader=WebBaseLoader(web_paths=("https://en.wikipedia.org/wiki/Paris",),)

docs=loader.load()
# print(docs)

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
docs=text_splitter.split_documents(docs)
vectorembeddings=Chroma.from_documents(docs,embedding=embeddings)
retriever=vectorembeddings.as_retriever(search_kwargs={"k":1})

system_prompt=("You are a helpful assistant that answers questions based on the provided context."
               "{context}")
prompt=ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")]
)
#without history aware retriever

# question_answer_chain=create_stuff_documents_chain(llm,prompt=prompt)
# rag_chain=create_retrieval_chain(retriever,question_answer_chain)

# response=rag_chain.invoke({"input":"What is the  population of  paris?"})
# print(response['answer'])

history_aware_retriever=create_history_aware_retriever(llm,retriever,prompt)
question_answer_chain=create_stuff_documents_chain(llm,prompt=prompt)
rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

chat_history=[]
while True:
    question=input("Ask a question: ")
    if question.lower() in ["exit", "quit"]:
        break
    response=rag_chain.invoke({"input":question,"chat_history":chat_history})
    chat_history.append((HumanMessage(content=question), AIMessage(content=response['answer'])))
    print(response['answer'])