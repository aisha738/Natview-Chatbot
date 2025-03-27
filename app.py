import streamlit as st
import requests
import openai
import pinecone
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Securely fetch API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment="us-west1-gcp")
index_name = "chatbot-index"
index = pinecone.Index(index_name)

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

 