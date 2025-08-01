# # phase3.py

# import os
# import warnings
# import logging
# from dotenv import load_dotenv
# import streamlit as st

# # LangChain Libraries
# from langchain_groq import ChatGroq
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.indexes import VectorstoreIndexCreator
# from langchain.chains import RetrievalQA

# # Load .env
# load_dotenv()

# warnings.filterwarnings("ignore")
# logging.getLogger("transformers").setLevel(logging.ERROR)

# st.title('Ask PDF - RAG Chatbot (Groq)')

# if 'messages' not in st.session_state:
#     st.session_state.messages = []

# # Show chat history
# for message in st.session_state.messages:
#     st.chat_message(message['role']).markdown(message['content'])

# # Load + Cache Vector Store from PDF
# @st.cache_resource
# def get_vectorstore():
#     pdf_name = "./data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"
#     loaders = [PyPDFLoader(pdf_name)]
#     index = VectorstoreIndexCreator(
#         embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
#         text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     ).from_loaders(loaders)
#     return index.vectorstore

# prompt = st.chat_input('Ask a question about the PDF...')

# if prompt:
#     st.chat_message('user').markdown(prompt)
#     st.session_state.messages.append({'role': 'user', 'content': prompt})

#     try:
#         vectorstore = get_vectorstore()
#         if vectorstore is None:
#             st.error("Failed to load PDF or vectorstore.")

#         # Setup Groq model
#         model = "llama3-8b-8192"
#         groq_chat = ChatGroq(
#             groq_api_key=os.environ.get("GROQ_API_KEY"), 
#             model_name=model
#         )

#         # Setup RetrievalQA
#         chain = RetrievalQA.from_chain_type(
#             llm=groq_chat,
#             chain_type='stuff',
#             retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
#             return_source_documents=True
#         )

#         # Get Answer
#         result = chain({"query": prompt})
#         response = result["result"]

#         # Display
#         st.chat_message('assistant').markdown(response)
#         st.session_state.messages.append({'role': 'assistant', 'content': response})

#     except Exception as e:
#         st.error(f"Error: {str(e)}")


import os
import warnings
import logging
from dotenv import load_dotenv
import streamlit as st

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.title('Ask PDF - RAG Chatbot (Groq)')

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Let user choose which vectorstore to use
available_dirs = os.listdir("vectorstores")
selected_db = st.selectbox("Choose a PDF to ask about:", available_dirs)

@st.cache_resource
def load_vectorstore(folder):
    path = os.path.join("vectorstores", folder)
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2')
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

if selected_db:
    vectorstore = load_vectorstore(selected_db)

    prompt = st.chat_input('Ask a question about the selected PDF...')


    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            # Setup Groq model
            model = "llama3-8b-8192"
            groq_chat = ChatGroq(
                groq_api_key=os.environ.get("GROQ_API_KEY"), 
                model_name=model
            )

            # Setup RetrievalQA
            chain = RetrievalQA.from_chain_type(
                llm=groq_chat,
                chain_type='stuff',
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True
            )

            result = chain({"query": prompt})
            response = result["result"].strip()

            # Reject hallucinated responses
            if not result["source_documents"] or response.lower().startswith("as an ai"):
                response = "❌ Sorry, I couldn't find anything relevant in the PDF."

            st.chat_message('assistant').markdown(response)
            st.session_state.messages.append({'role': 'assistant', 'content': response})

        except Exception as e:
            st.error(f"Error: {str(e)}")
