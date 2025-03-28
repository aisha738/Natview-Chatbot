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
import time

# Retrieve API keys from Streamlit Secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index_name = "langchain-test-index"
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

# Streamlit UI Customization
st.markdown("""
    <style>
        .title {
            text-align: center;
            color: blue;
            font-size: 36px;
            font-weight: bold;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: black;
        }
        .stTextInput {
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>NatBot</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Your AI Companion for NFTI</div>", unsafe_allow_html=True)

# Sidebar for search history
st.sidebar.header("Search History")
for past_query in st.session_state['search_history']:
    st.sidebar.write(past_query)

user_input = st.text_input("Ask a question:")
use_web_search = st.checkbox("Use live web search if needed")

if user_input:
    st.session_state['search_history'].append(user_input)
    
    pinecone_response = chatbot_response(user_input)
    
    if pinecone_response:
        response = pinecone_response
    elif use_web_search:
        response = search_web(user_input)
    else:
        response = "No relevant information found."
    
    st.markdown("### 🔍 Response:")
    st.write(response)

