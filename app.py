import os
import time
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

# Define the correct index name
index_name = "index"
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# Initialize Tavily search tool
search_tool = TavilySearchResults(api_key=TAVILY_API_KEY)

# Initialize session state for search history
toggle_sidebar = st.sidebar.checkbox("📜 Show Search History")
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
st.markdown("<h1 style='text-align: center; color: blue;'>NatBot</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; font-size: 16px;'>Your AI Companion for NFTI – Information at Your Fingertips</h4>", unsafe_allow_html=True)

st.markdown("""
    <style>
        .stButton>button {width: 50%; margin: auto; display: block; background-color: blue; color: white;}
        .stTextInput>div>div>input {margin-top: 20px;}
    </style>
""", unsafe_allow_html=True)

# Sidebar for search history
if toggle_sidebar:
    st.sidebar.header("Search History")
    for past_query in st.session_state['search_history']:
        st.sidebar.write(past_query)

user_input = st.text_input("Ask a question:")
use_web_search = st.checkbox("Use live web search if needed")
search_button = st.button("Search")

if search_button and user_input:
    st.session_state['search_history'].append(user_input)  # Store query in history

    # Retrieve from Pinecone first
    pinecone_response = chatbot_response(user_input)
    
    if pinecone_response:  # If Pinecone returns relevant data
        st.markdown("### ✅ Database Match Found!")
        response = pinecone_response
    else:  # If no relevant data, fall back to Web Search
        st.markdown("### 🌐 No match found in the database, searching online...")
        response = search_web(user_input)

    # Display results in a structured and direct format
    st.markdown("### 🔍 Answer:")
    st.write(response)


