import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun,DuckDuckGoSearchResults
from langchain.agents import initialize_agent, AgentType    
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv


arxiv_wrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)
wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=wikipedia_wrapper)
ddg = DuckDuckGoSearchResults(name='Search')
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

st.title("Chat with Search Tools")

if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {"role": "assistant", "content": """You are a smart AI assistant.

Before using any tools, ask yourself: 
- Can I answer this confidently without tools?
- Is the question simple (e.g., greetings, basic math, common facts)?

If yes, answer directly.

Use only one tool unless absolutely necessary.

Prefer the most relevant tool for the question:
- For general definitions → use Wikipedia
- For research papers → use Arxiv
- For real-time updates → use DuckDuckGo

Avoid redundant tool calls unless information is insufficient.
"""
}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask a question:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(model_name="llama-3.3-70b-versatile", streaming=True)
    tools = [arxiv, wiki, ddg]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.SELF_ASK_WITH_SEARCH,
        verbose=True,
        handle_parsing_errors=True
    )
    with st.chat_message("assistant"):
        st_cb= StreamlitCallbackHandler(st.container(),expand_new_thoughts=True)
        response = agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)