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

corrie = "im a hoe" 
richness = "aku orang batak"
keno = "iamkeno"
rafsan = "mysuguru<3" 

st.header('Login first!!!')

st.write("What is the password?", key = "password")
password = st.text_input("")

if password == richness or password == corrie or password == keno : 

    if password == corrie :
        st.title("HAPPY BIRTHDAY LGBT BITCH!!! :unamused: :unamused: :rainbow: :rainbow: :rainbow:")
        st.subheader("YOU'RE FUCKING FREE :rainbow: :rainbow: :rainbow:")
        st.write("I call dibs that this is the most useful gift you will receive today")

    if password == richness :
        st.write("Hey Richness!! you are very cool and this is a gift for you")
        st.write("Hopefully it will help you in your studies :grin: :grin:")

    if password == rafsan :
        st.title("hello pookie :rainbow: :rainbow: :rainbow:")
        st.subheader("are you gay? :rainbow: :rainbow: :rainbow:")

    if password == richness : 
        client = Groq(api_key="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")
    if password == corrie : 
        client = Groq(api_key="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")
    if password == keno :
        client = Groq(api_key="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")
    if password == rafsan : 
        client = Groq(api_key="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")
    

    filename = ""

    if 'pdf_ref' not in ss:
        ss.pdf_ref = None

    with st.sidebar :
        st.subheader('Please input file', divider='rainbow')
        st.write("""Before chatting, please :blue[upload the txt file first] and provide context, or :blue[input the text as context inside the "Input text from above here"] section below, else the bot will be confused :slightly_frowning_face: :slightly_frowning_face:""")
        st.write(":blue[NOTE : The maximum is 5250 words or 27500 characters], please check before inputting into system, else it will go KABOOM")
        st.header("")

    with st.sidebar : 
        st.file_uploader("Upload .txt file", type='txt', key='pdf')
        st.write("The engine only accepts .txt file, please only upload txt files")
        st.header("")
        st.write("If you haven't converted it into .txt, convert it through the website here :")
        st.code("pdf2go.com/pdf-to-text")
        st.header("")
        st.header("")
        if ss.pdf:
            ss.pdf_ref = ss.pdf
        
    if ss.pdf_ref != None :
        binary_data = ss.pdf_ref.getvalue()
    if ss.pdf_ref == None : 
        binary_data = ""

    with st.sidebar : 
        st.subheader('Text content of .txt file', divider='rainbow')
        st.code(binary_data) 
        st.write("Copy the text here into the text input below")
        st.header("")

    binary_data = str(binary_data)

    with st.sidebar : 
        st.subheader('Input text from above here', divider='rainbow')
        data = st.text_input("Input Text")
        st.write("Note : Copy the text above to here")

    data = str(data)
    #data = data.replace("\r\", "")

    with st.sidebar : 
        st.header("")
        st.subheader('Smart Summary', divider='rainbow')

    if data != "" : 
        summary_bullet_point = f"Summarize {data} into 10 bullet points, just print the bullet points, don't add anything else, not even an introduction"
        bulletpointsummary = client.chat.completions.create(messages=[{"role":"user", "content":summary_bullet_point,}],model="llama3-8b-8192")
        bulletpointsummary =  bulletpointsummary.choices[0].message.content
    
    with st.sidebar : 
        if data != "" : 
            st.code(bulletpointsummary)
    
    # STREAMLIT CODE -------------------------------------------------------------
    
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

        if password == richness :
            question_answer = f"Answer the question of {prompt} only from the context of {data} for multiple choice questions, just answer and give only 1 sentence detail"
            finalanswer = client.chat.completions.create(messages=[{"role":"user", "content":question_answer,}],model="llama3-8b-8192")
            finalanswer =  finalanswer.choices[0].message.content

        if password == keno :
            question_answer = f"Answer the question of {prompt} only from the context of {data} and answer explain in detail"
            finalanswer = client.chat.completions.create(messages=[{"role":"user", "content":question_answer,}],model="llama3-8b-8192")
            finalanswer =  finalanswer.choices[0].message.content
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(finalanswer)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": finalanswer})
