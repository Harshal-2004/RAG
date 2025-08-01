# phase2.py

import warnings
import logging
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Disable warnings and info logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.title('Ask Chatbot (Groq LLM)')
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Get user prompt
prompt = st.chat_input('Pass your prompt here')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # Setup Prompt
    groq_sys_prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Answer clearly and concisely. 
    Question: {user_prompt}
    """)

    # Use Groq's Llama3 model
    model = "llama3-8b-8192"
    groq_chat = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=model
    )

    # Combine components into chain
    chain = groq_sys_prompt | groq_chat | StrOutputParser()

    # Run model
    response = chain.invoke({"user_prompt": prompt})

    # Display and store response
    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role': 'assistant', 'content': response})
