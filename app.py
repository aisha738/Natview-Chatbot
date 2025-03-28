import os
import time
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain.embeddings.base import Embeddings
import pinecone
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
import requests
import json
from langchain.llms import LLM
#from langchain_core.llms import LLM
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun

# Retrieve API keys from Streamlit Secrets
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Define the correct index name
index_name = "langchain-test-index"
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# Initialize Google Gemini client for embeddings
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Define your embeddings class
class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document",
                title="document"
            )
            embeddings.append(result["embedding"])
        return embeddings

    def embed_query(self, text):
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query",
            title="query"
        )
        return result["embedding"]

# Initialize the retriever using Pinecone
def get_retriever():
    embeddings = GeminiEmbeddings()
    vectorstore = PineconeVectorStore.from_existing_index(index_name, embeddings)
    return vectorstore.as_retriever()

# Groq API function to get chatbot responses
def groq_chatbot_response(query):
    url = "https://api.groq.ai/v1/generate"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "input": query,
        "model": "llama2-70b-4096",
        "temperature": 0.7,
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code} - {response.text}"

# Define a wrapper class to use Groq as an LLM
class GroqLLM(LLM):
    def __init__(self, api_key):
        self.api_key = api_key

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any,) -> str:
        return groq_chatbot_response(prompt)

    @property
    def _identifying_params(self) -> dict:
        return {"groq_api_key": self.api_key}

    @property
    def _llm_type(self) -> str:
        return "groq"

# Chatbot response logic using Pinecone + Groq
def chatbot_response(query):
    # First, try to retrieve from Pinecone (database)
    retriever = get_retriever()
    llm = GroqLLM(api_key=GROQ_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    pinecone_response = qa_chain.run(query)

    if pinecone_response:
        return pinecone_response
    else:
        return groq_chatbot_response(query)

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: blue;'>NatBot</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; font-size: 16px;'>Your AI Companion for NFTI – Information at Your Fingertips</h4>", unsafe_allow_html=True)

# Initialize session state for search history
if 'search_history' not in st.session_state:
    st.session_state['search_history'] = []
if 'sidebar_state' not in st.session_state:
    st.session_state['sidebar_state'] = False

# Sidebar for search history
with st.sidebar:
    if st.button("📜Search History"):
        st.session_state['sidebar_state'] = not st.session_state['sidebar_state']
    if st.session_state['sidebar_state']:
        st.sidebar.header("Search History")
        for past_query in st.session_state['search_history']:
            st.sidebar.write(past_query)

user_input = st.text_input("Ask a question:")
search_button = st.button("Search")

if search_button and user_input:
    st.session_state['search_history'].append(user_input)

    response = chatbot_response(user_input)

    st.markdown("### 🔍 Answer:")
    st.write(response)
