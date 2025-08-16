import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

st.title("PDF to Vector DB (Phase 1)")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    filename = uploaded_file.name
    base_filename = os.path.splitext(filename)[0]
    vectorstore_dir = os.path.join("vectorstores", base_filename)

    os.makedirs("data", exist_ok=True)
    os.makedirs("vectorstores", exist_ok=True)

    file_path = os.path.join("data", filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2')  

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(vectorstore_dir)

    st.success(f"Vector DB created and saved to: {vectorstore_dir}")
