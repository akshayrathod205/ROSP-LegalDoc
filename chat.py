from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key=os.getenv('GROQ_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

## Prompt Template

llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_messages(
    [
        ("system",'''You are a specialized legal assistant trained in the Indian legal context. 
         You should only answer questions related to legal matters, acts, laws, and the Constitution of India. 
         Do not provide responses to non-legal queries. Your goal is to provide accurate, concise, and informative legal answers based strictly on the user's input. 
         Only respond to legal queries with factual and well-structured information. If the question is outside the legal context, politely decline to answer and inform the user that you only handle legal inquiries. 
         When responding, include references to relevant laws or sections of the Indian Constitution if applicable.'''),
        ("user","Question:{question}")
    ]
)
## streamlit framework

st.title('Chat with Legal Assistant')
input_text=st.text_input("How can I help you today?")

output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))