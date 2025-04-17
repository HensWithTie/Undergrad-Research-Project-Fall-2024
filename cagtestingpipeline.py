from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import TextLoader #USE TEXTLOADER
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata

#CAG IMPORTS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import time

with open("Jon.txt", "r", encoding="utf-8") as f:
    doc_text = f.read()

prompt = "What does Jonathan like?"

system_prompt = f"""
<|system|>
You are an assistant who provides concise factual answers. You will read the text file
and respond with a max of 3 sentences. If you do not know, say "I don't know".
<|user|>
Context:
{doc_text}
Question:
{prompt}
""".strip()

model_client = ChatOllama(model='llama3.1')
response = model_client.invoke(system_prompt)
print(response)
