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
index_name = "langchain-test-index"  # Correct index name

# Ensure the index exists before using it
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    st.warning("⚠️ Index not found! Creating a new Pinecone index...")
    pc.create_index(
        name=index_name,
        dimension=3072,  # Correct dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # Correct region
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

# Use the existing index
index = pc.Index(index_name)

# Initialize Tavily search tool
search_tool = TavilySearchResults(api_key=TAVILY_API_KEY)

# Initialize session state for search history
if 'search_history' not in st.session_state:
    st.session_state['search_history'] = []

# Function to retrieve from Pinecone
def get_retriever():
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = Pinecone(index, embeddings)  # Use the `index` variable
        return vectorstore.as_retriever()
    except Exception as e:
        st.error(f"🚨 Error loading index: {e}")
        return None

# Function to get chatbot response
def chatbot_response(query):
    retriever = get_retriever()
    if retriever:
        qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)
        return qa_chain.run(query)
    else:
        return None

# Function to search the web
def search_web(query):
    return search_tool.run(query)

# Streamlit UI
st.title("NatBot")
st.subheader("Your AI Companion for NFTI – Information at Your Fingertips")

# Sidebar for search history
st.sidebar.header("Search History")
for past_query in st.session_state['search_history']:
    st.sidebar.write(past_query)

user_input = st.text_input("Ask a question:")
use_web_search = st.checkbox("Use live web search if needed")

if user_input:
    st.session_state['search_history'].append(user_input)  # Store query in history

    # Retrieve from Pinecone first
    pinecone_response = chatbot_response(user_input)

    if pinecone_response:  # If Pinecone returns relevant data
        st.markdown("### ✅ Database Match Found!")
        response = pinecone_response
    else:  # If no relevant data, fall back to Web Search
        st.markdown("### 🌐 No match found in the database, searching online...")
        response = search_web(user_input)

    # Display results in a structured and colorful format
    st.markdown("### 🔍 Here’s what I found:")

    if isinstance(response, list):  # Handle structured web search responses
        for idx, result in enumerate(response):
            st.markdown(f"""
            **{idx+1}. {result['title']}**  
            🌐 [Visit Website]({result['url']})  
            📌 **Relevance Score:** `{result['score']:.2f}`  
            📝 **Summary:** {result['content'][:300]}...  
            """, unsafe_allow_html=True)
    else:  # Handle direct text responses from Pinecone
        st.markdown(response)

