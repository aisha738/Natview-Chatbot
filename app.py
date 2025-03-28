import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain.tools.tavily_search import TavilySearchResults

# Retrieve API keys from Streamlit Secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index_name = "langchain-test-index"

# Initialize Tavily search tool
search_tool = TavilySearchResults(api_key=TAVILY_API_KEY)

# Initialize session state for search history
if 'search_history' not in st.session_state:
    st.session_state['search_history'] = []

def get_retriever():
    embeddings = OpenAIEmbeddings()
    vectorstore = Pinecone.from_existing_index(index_name, embeddings)
    return vectorstore.as_retriever()

def chatbot_response(query):
    retriever = get_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)
    return qa_chain.run(query)

def search_web(query):
    return search_tool.run(query)

# Streamlit UI
st.markdown("""
    <h1 style='text-align: center; color: blue;'>NatBot</h1>
    <h4 style='text-align: center; color: black; font-size: 16px;'>Your AI Companion for NFTI</h4>
""", unsafe_allow_html=True)

st.write("""
    <style>
        .stTextInput>div>div>input {
            text-align: center;
        }
        .stButton>button {
            display: block;
            margin: 0 auto;
            background-color: blue !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

user_input = st.text_input("", placeholder="Ask a question...", key="user_input")
search_btn = st.button("Search")

if search_btn and user_input:
    st.session_state['search_history'].append(user_input)
    pinecone_response = chatbot_response(user_input)
    
    if pinecone_response:
        response = pinecone_response
    else:
        response = search_web(user_input)
    
    st.markdown(f"**Response:** {response}")


