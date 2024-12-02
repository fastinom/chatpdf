import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# Groq API Endpoints
GROQ_TEXT_GEN_URL = "https://api.groq.com/text-gen"
GROQ_EMBEDDING_URL = "https://api.groq.com/embedding"
GROQ_IMAGE_GEN_URL = "https://api.groq.com/image-gen"
GROQ_VECTOR_DB_URL = "https://api.groq.com/vector-search"

# Helper functions
def generate_text(prompt, context=""):
    """Generate text using Groq API."""
    payload = {"prompt": prompt, "context": context, "max_length": 200}
    response = requests.post(GROQ_TEXT_GEN_URL, json=payload)
    response.raise_for_status()
    return response.json().get("text", "No response")

def get_relevant_docs(prompt, top_k=3):
    """Retrieve documents using Groq API."""
    payload = {"query": prompt, "top_k": top_k}
    response = requests.post(GROQ_VECTOR_DB_URL, json=payload)
    response.raise_for_status()
    return response.json().get("documents", [])

def generate_image(prompt):
    """Generate an image using Groq API."""
    payload = {"prompt": prompt}
    response = requests.post(GROQ_IMAGE_GEN_URL, json=payload)
    response.raise_for_status()
    image_bytes = BytesIO(response.content)
    return Image.open(image_bytes)

# Streamlit App
st.title("RAG with Groq APIs")

# Input prompt
user_prompt = st.text_input("Enter your prompt:", placeholder="E.g., 'Explain climate change with visuals'")

if st.button("Generate"):
    if user_prompt:
        with st.spinner("Retrieving relevant documents..."):
            docs = get_relevant_docs(user_prompt)
            retrieved_texts = [doc['content'] for doc in docs]
        
        context = " ".join(retrieved_texts)

        with st.spinner("Generating text response..."):
            text_response = generate_text(user_prompt, context)
        
        st.subheader("Generated Text")
        st.write(text_response)

        with st.spinner("Generating relevant images..."):
            image = generate_image(user_prompt)
        
        st.subheader("Generated Image")
        st.image(image, caption="Generated Visual", use_column_width=True)
    else:
        st.warning("Please enter a prompt to generate a response.")

