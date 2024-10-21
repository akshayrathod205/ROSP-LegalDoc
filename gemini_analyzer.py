import streamlit as st
import PyPDF2 as PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pdf2image import convert_from_path
import pytesseract
import os

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Specify the path to your Tesseract installation (modify this path as needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader.PdfReader(pdf)
        for page in pdf_reader.pages:
            # Check if the page is image-based or text-based
            if '/XObject' in page['/Resources']:
                # If image-based, convert to image and apply OCR
                images = convert_from_path(pdf, first_page=page)
                for img in images:
                    text += pytesseract.image_to_string(img)
            else:
                # If text-based, extract text directly
                text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks

def build_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def get_conversational_chain():
    prompt_template = """
    You are a specialized legal assistant trained to handle legal questions. 
    Your job is to provide clear, accurate, and concise legal advice strictly based on the context or question provided by the user. 
    Your responses must be grounded in legal matters, such as laws, regulations, contracts, and legal principles. 
    If the user’s question falls outside the legal domain or involves content not directly related to legal concerns, politely decline to answer, explaining that you only provide legal information.
    Only respond to questions where you can give relevant legal insights, including references to applicable laws or principles when necessary.
    Do not provide responses or interpretations of general documents or PDFs unless the question directly asks about a legal issue within them.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, vector_store):
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain.invoke(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    print(response)
    st.write("Reply: ", response["output_text"])
    
def main():
    st.set_page_config("Chat with PDF")
    st.header("Chat with PDF using Gemini") 
    user_question = st.text_input("Ask a Question from the PDF Files")
    if 'vector_store' in st.session_state:
        if user_question:
            user_input(user_question, st.session_state['vector_store'])
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = build_vector_store(text_chunks)
                st.session_state['vector_store'] = vector_store
                st.success("Done")
                
if __name__ == "__main__":
    main()
