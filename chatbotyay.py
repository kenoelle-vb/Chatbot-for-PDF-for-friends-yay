import langchain
from groq import Groq
import sentence_transformers
from langchain_community.document_loaders import TextLoader
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

loader = TextLoader("macroman2.txt")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=1000)
chunks = text_splitter.split_documents(data)

llm = ChatGroq(groq_api_key="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", model_name="llama3-8b-8192")

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

documents = chunks
embeddings = FastEmbedEmbeddings()
vector_store = Chroma.from_documents(documents, embeddings)

prompting = PromptTemplate(
    input_variables=["question"],
    template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context
    to answer the question, please answer the question, give as much data as possible
    regarding the context of information, make connections with other points, and elongate the sentence"""
        )

retriever = MultiQueryRetriever.from_llm(vector_store.as_retriever(), llm, prompting)

template = "Answer only from the following context : {context}, Question:{question}"
prompting = ChatPromptTemplate.from_template(template)

chain = ({"context": retriever, "question": RunnablePassthrough()}
                      | prompting
                      | llm
                      | StrOutputParser())

# STREAMLIT CODE -------------------------------------------------------------

st.title("Chat Bot")

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

    response = chain.invoke(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})