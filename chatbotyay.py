import pdfminer
from pypdf import PdfReader
import langchain
from groq import Groq
import sentence_transformers
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate, PromptTemplate
import streamlit as st
import tempfile
from streamlit import session_state as ss
from streamlit_pdf_viewer import pdf_viewer

st.header('Login first!!!')

st.write("Who are you?")
name = st.text_input("", key = "name")
st.write("What is the password?", key = "password")
password = st.text_input("")

if name == "Richness" and password == "akuorangbatak" : 

    filename = ""

    if 'pdf_ref' not in ss:
        ss.pdf_ref = None

    with st.sidebar : 
        st.file_uploader("Upload .txt file", type='txt', key='pdf')
        if ss.pdf:
            ss.pdf_ref = ss.pdf
        
    if ss.pdf_ref:
        binary_data = ss.pdf_ref.getvalue()

    with st.sidebar : 
        st.code(binary_data) 

    binary_data = str(binary_data)

    with st.sidebar : 
        data = st.text_input("Input Text", "Please Input Text")

    data = str(data)
    #data = data.replace("\r\", "")

    summary_bullet_point = f"Summarize {data} into 10 bullet points, just print the bullet points, don't add anything else, not even an introduction:"
    bulletpointsummary = client.chat.completions.create(messages=[{"role":"user", "content":summary_bullet_point,}],model="llama3-8b-8192")

    with st.sidebar : 
        st.code(bulletpointsummary)
    
    # STREAMLIT CODE -------------------------------------------------------------

    client = Groq(api_key="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")
    
    st.title("Chat Bot")
    st.write("by Keno 4 u") 
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        question_answer = f"Answer the question of {prompt} only from the context of {data}"
        finalanswer = client.chat.completions.create(messages=[{"role":"user", "content":question_answer,}],model="llama3-8b-8192")
    
        response = chain.invoke(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
