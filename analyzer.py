import streamlit as st
import os
import io
import PyPDF2
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.schema import Document  # Import the Document class
from dotenv import load_dotenv


load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

st.title("Legal Document Analysis")

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    You are a specialized legal assistant trained to handle legal questions. 
    Your job is to provide clear, accurate, and concise legal advice strictly based on the context or question provided by the user. 
    Your responses must be grounded in legal matters, such as laws, regulations, contracts, and legal principles. 
    If the user’s question falls outside the legal domain or involves content not directly related to legal concerns, politely decline to answer, explaining that you only provide legal information.
    Only respond to questions where you can give relevant legal insights, including references to applicable laws or principles when necessary.
    Do not provide responses or interpretations of general documents or PDFs unless the question directly asks about a legal issue within them.\n\n
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Function to handle PDF embedding
def vector_embedding(pdf_file):
    if "vectors" not in st.session_state:
        # Convert the UploadedFile to a file-like object using io.BytesIO
        pdf_file_bytes = io.BytesIO(pdf_file.read())
        
        # Use PyPDF2 to extract text from the PDF file
        pdf_reader = PyPDF2.PdfReader(pdf_file_bytes)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Split the extracted text into documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_text(text)
        
        # Convert each chunk of text into a Document object
        documents = [Document(page_content=doc) for doc in final_documents]
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.vectors = FAISS.from_documents(documents, embeddings)

# File uploader for manual PDF upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    if st.button("Submit"):
        vector_embedding(uploaded_file)
        st.write("Vector Store DB is ready.")

# Accept user input
prompt1 = st.text_input("How can I help?")

import time

if prompt1 and "vectors" in st.session_state:
    # Create the chain for document retrieval and LLM responses
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.write(f"Response time: {time.process_time() - start} seconds")
    
    st.write(response['answer'])

    # Display the relevant document sections in an expander
    with st.expander("Retrieve Relevant Sections"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
