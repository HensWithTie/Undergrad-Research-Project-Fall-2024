from langchain_community.chat_models import ChatOllama

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
print("Llama3.1 loaded!")

stime = time.time()

response = model_client.invoke(system_prompt)
print(response)

etime = time.time()
ttime = etime - stime
print(f"total time: {ttime:.4f} seconds")