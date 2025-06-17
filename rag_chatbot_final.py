import os
import streamlit as st
import warnings
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title="Ask Chatbot!", layout="centered")
st.title("Ask Chatbot! 🤖")

PDF_PATH = "reflexion.pdf"

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

@st.cache_resource
def load_vectorstore():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError("PDF not found in the folder.")
    loader = PyPDFLoader(PDF_PATH)
    return VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2"),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders([loader]).vectorstore

prompt = st.chat_input("Ask anything about the PDF...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        vectorstore = load_vectorstore()
        llm = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"], model_name="llama3-8b-8192")
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        result = chain({"query": prompt})
        response = result["result"]
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"🚫 Error: {str(e)}")
