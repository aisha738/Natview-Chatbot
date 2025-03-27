import streamlit as st
from pinecone import Pinecone
import openai
import requests
from typing import Optional, Dict, List

# ======================
# APP CONFIGURATION
# ======================
st.set_page_config(page_title="Natview AI Chatbot", page_icon="🤖")

# ======================
# INITIALIZATION
# ======================
def initialize_services() -> Optional[Dict]:
    """Initialize all API services with proper error handling"""
    try:
        required_keys = {
            'PINECONE_API_KEY': 'Pinecone',
            'GROQ_API_KEY': 'Groq',
            'OPENAI_API_KEY': 'OpenAI',
            'SERPER_API_KEY': 'Serper'
        }
        
        missing = [name for key, name in required_keys.items() if key not in st.secrets]
        if missing:
            st.error(f"Missing secrets for: {', '.join(missing)}")
            return None

        # Initialize Pinecone
        pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        index_name = "chatbot-index"
        
        if index_name not in pc.list_indexes():
            st.error(f"Pinecone index '{index_name}' not found.")
            return None
        
        index = pc.Index(index_name)
        
        # Test Pinecone connection
        try:
            index.describe_index_stats()
        except Exception as e:
            st.error(f"Pinecone connection failed: {str(e)}")
            return None

        openai.api_key = st.secrets["OPENAI_API_KEY"]
        
        return {
            'pinecone_index': index,
            'groq_key': st.secrets["GROQ_API_KEY"],
            'serper_key': st.secrets["SERPER_API_KEY"]
        }
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        return None

# ======================
# CORE FUNCTIONS
# ======================
def get_chatbot_response(query: str, api_key: str) -> str:
    """Get response from Groq's LLaMA model"""
    try:
        response = requests.post(
            "https://api.groq.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-8b",
                "messages": [{"role": "user", "content": query}]
            },
            timeout=10
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"⚠️ API Error: {str(e)}"

def get_openai_embedding(query: str) -> Optional[List[float]]:
    """Generate embeddings using OpenAI"""
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=query
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        st.error(f"Embedding error: {str(e)}")
        return None

def search_pinecone(index, embedding: List[float]) -> str:
    """Query Pinecone vector database"""
    try:
        result = index.query(
            vector=embedding,
            top_k=1,
            include_metadata=True
        )
        return result["matches"][0]["metadata"]["text"] if result["matches"] else "No relevant results found."
    except Exception as e:
        return f"⚠️ Pinecone error: {str(e)}"

def search_web(query: str, api_key: str) -> str:
    """Get search results from Serper"""
    try:
        response = requests.get(
            "https://serper.dev/search",
            params={"q": query},
            headers={"X-API-KEY": api_key},
            timeout=10
        )
        response.raise_for_status()
        results = response.json().get("organic", [])[:3]
        return "\n".join(f"• [{res['title']}]({res['link']})" for res in results)
    except requests.exceptions.RequestException as e:
        return f"⚠️ Search error: {str(e)}"

# ======================
# STREAMLIT UI
# ======================
def main():
    st.title("Natview AI Chatbot")
    st.caption("Powered by Groq, Pinecone, OpenAI, and Serper")

    services = initialize_services()
    if not services:
        st.stop()

    with st.form("chat_form"):
        query = st.text_input("Ask me anything:", placeholder="Type your question...")
        submitted = st.form_submit_button("Submit")

    if submitted and query:
        with st.spinner("Generating response..."):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("AI Response")
                response = get_chatbot_response(query, services['groq_key'])
                st.markdown(response)

            with col2:
                st.subheader("Knowledge Base")
                embedding = get_openai_embedding(query)
                if embedding:
                    pinecone_result = search_pinecone(services['pinecone_index'], embedding)
                    st.markdown(pinecone_result)

            st.divider()
            with st.expander("🌐 Web Search Results"):
                st.markdown(search_web(query, services['serper_key']))

if __name__ == "__main__":
    main()
