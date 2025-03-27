import streamlit as st
from pinecone import Pinecone
import os
from dotenv import load_dotenv

# Initialize with error handling
try:
    # Try Streamlit secrets first (for deployment)
    pinecone_key = st.secrets.get("PINECONE_API_KEY")
    
    # Fallback to .env for local development
    if not pinecone_key:
        load_dotenv()
        pinecone_key = os.getenv("PINECONE_API_KEY")
    
    if not pinecone_key:
        st.error("🚨 Pinecone API key not found! Check your secrets/config.")
        st.stop()
    
    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index("chatbot-index")
    
    # Test connection
    index.describe_index_stats()
    st.success("✅ Successfully connected to Pinecone!")
    
except Exception as e:
    st.error(f"❌ Pinecone initialization failed: {str(e)}")
    st.stop()

# Query example (v3+ syntax)
index.query(vector=[...], top_k=5)

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# Function to get chatbot response using Groq
def get_chatbot_response(query):
    url = "https://api.groq.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {"model": "llama3-8b", "messages": [{"role": "user", "content": query}]}
    response = requests.post(url, json=data, headers=headers)
    return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response")

# Function to perform web search using Serper
def search_web(query):
    search_url = f"https://serper.dev/search?q={query}&api_key={SERPER_API_KEY}"
    response = requests.get(search_url)
    results = response.json().get("organic_results", [])
    return "\n".join([f"- {r['title']}: {r['link']}" for r in results[:3]])

# Function to get OpenAI embedding for the query
def get_openai_embedding(query):
    response = openai.Embedding.create(model="text-embedding-ada-002", input=query)
    return response['data'][0]['embedding']

# Function to search Pinecone index using the embedding
def search_pinecone(query_embedding):
    result = index.query([query_embedding], top_k=1, include_values=True)
    if result['matches']:
        return result['matches'][0]['metadata']['text']  # Adjust according to your metadata structure
    return "No relevant information found in Pinecone."

# Streamlit app interface
def main():
    st.title("Natview AI Chatbot")
    st.write("An AI chatbot powered by Groq, Pinecone, OpenAI, and Serper.")

    user_input = st.text_input("Ask me anything:")
    if st.button("Send"):
        # Get chatbot response from Groq
        response = get_chatbot_response(user_input)
        st.write("**Chatbot (Groq):**", response)

        # Get OpenAI embedding for the query
        query_embedding = get_openai_embedding(user_input)

        # Search Pinecone for relevant information
        pinecone_result = search_pinecone(query_embedding)
        st.write("**Pinecone Search Result:**", pinecone_result)

        # Perform web search for related information using Serper
        st.write("\n🔎 **Related Web Search Results (Serper):**")
        web_results = search_web(user_input)
        st.write(web_results)

if __name__ == "__main__":
    main()

 
