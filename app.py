import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.tools.tavily_search import TavilySearchResults

# Retrieve API keys from Streamlit Secrets
# GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
# OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
#PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
#PINECONE_ENV = st.secrets["PINECONE_ENV"]
#TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]

PINECONE_API_KEY="pc4BAwuS_MvNfWZwdAa72JZQBYPDC3ZX98Mjtozv5UppsZx3CkUMrbNosF8n6vo9xxG5awXS"
GROQ_API_KEY="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"
OPENAI_API_KEY="sk-proj-iXWVY2y83TgqfrrSEW8hdh73SDKPZ8NzohuIZM5WrR4Uzl5PUqFuWiXB20hWRjXQ1AAOuoGcWiT3BlbkFJvHJ4wtrk4HzGKF3JOlA_VhWKErmUvXIkgmAefPnNR8lZrRmY2gZMaPecIZjgJpVAti27MA"
SERPER_API_KEY="fee3f70571c3c4be9d5d3e193e0946fa169c6405"
TAVILY_API_KEY="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"
PINECONE_ENV="us-east-1-aws"

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Initialize Tavily search tool
search_tool = TavilySearchResults(api_key=TAVILY_API_KEY)

# Initialize session state for search history
if 'search_history' not in st.session_state:
    st.session_state['search_history'] = []

def get_retriever():
    embeddings = OpenAIEmbeddings()
    vectorstore = Pinecone.from_existing_index("your_index_name", embeddings)
    return vectorstore.as_retriever()

def chatbot_response(query):
    retriever = get_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)
    return qa_chain.run(query)

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
    if use_web_search:
        response = search_web(user_input)  # Use Tavily for live search
    else:
        response = chatbot_response(user_input)  # Use Pinecone retrieval
    
    st.write("Response:", response)
