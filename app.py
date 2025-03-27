import streamlit as st
from pinecone import Pinecone
import openai
import os
import requests
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()  # Load .env file for local development

# Initialize all API keys with proper error handling
try:
    # Pinecone
    pinecone_key = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")
    if not pinecone_key:
        st.error("Missing Pinecone API Key! Add to Streamlit Secrets or .env file")
        st.stop()
    
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index("chatbot-index")
    
    # Test Pinecone connection
    try:
        index.describe_index_stats()
    except Exception as e:
        st.error(f"Pinecone connection failed: {str(e)}")
        st.stop()

    # Other API keys
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    SERPER_API_KEY = st.secrets.get("SERPER_API_KEY") or os.getenv("SERPER_API_KEY")
    
    openai.api_key = OPENAI_API_KEY

except Exception as e:
    st.error(f"Initialization error: {str(e)}")
    st.stop()

# --- Core Functions ---
def get_chatbot_response(query):
    """Get response from Groq API"""
    try:
        url = "https://api.groq.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama3-8b",
            "messages": [{"role": "user", "content": query}]
        }
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response")
    except Exception as e:
        return f"Error getting Groq response: {str(e)}"

def search_web(query):
    """Search web using Serper API"""
    try:
        search_url = f"https://serper.dev/search?q={query}&api_key={SERPER_API_KEY}"
        response = requests.get(search_url)
        response.raise_for_status()
        results = response.json().get("organic_results", [])
        return "\n".join([f"- {r['title']}: {r['link']}" for r in results[:3]])
    except Exception as e:
        return f"Error performing web search: {str(e)}"

def get_openai_embedding(query):
    """Get embeddings from OpenAI"""
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=query
        )
        return response['data'][0]['embedding']
    except Exception as e:
        st.error(f"Error getting embeddings: {str(e)}")
        return None

def search_pinecone(query_embedding):
    """Search Pinecone index"""
    try:
        result = index.query(
            vector=query_embedding,
            top_k=1,
            include_metadata=True
        )
        if result['matches']:
            return result['matches'][0]['metadata']['text']
        return "No relevant information found in Pinecone."
    except Exception as e:
        return f"Error searching Pinecone: {str(e)}"

# --- Streamlit UI ---
def main():
    st.title("Natview AI Chatbot")
    st.write("An AI chatbot powered by Groq, Pinecone, OpenAI, and Serper.")

    user_input = st.text_input("Ask me anything:")
    if st.button("Send") and user_input:
        with st.spinner("Processing..."):
            # Get chatbot response
            response = get_chatbot_response(user_input)
            st.write("**Chatbot (Groq):**", response)

            # Get and search embeddings
            query_embedding = get_openai_embedding(user_input)
            if query_embedding:
                pinecone_result = search_pinecone(query_embedding)
                st.write("**Pinecone Search Result:**", pinecone_result)

            # Web search
            st.write("\n🔎 **Related Web Search Results (Serper):**")
            web_results = search_web(user_input)
            st.write(web_results)

if __name__ == "__main__":
    main()

 
