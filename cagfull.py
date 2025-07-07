import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import os
import time
from accelerate import disk_offload

def generate(model, input_ids: torch.Tensor, past_key_values, max_new_tokens: int = 50) -> torch.Tensor:
    device = model.model.embed_tokens.weight.device
    origin_len = input_ids.shape[-1]
    input_ids = input_ids.to(device)
    output_ids = input_ids.clone()
    next_token = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = out.logits[:, -1, :]
            token = torch.argmax(logits, dim=-1, keepdim=True)
            output_ids = torch.cat([output_ids, token], dim=-1)
            past_key_values = out.past_key_values
            next_token = token.to(device)

            if model.config.eos_token_id is not None and token.item() == model.config.eos_token_id:
                break
    return output_ids[:, origin_len:]

def get_kv_cache(model, tokenizer, prompt: str) -> DynamicCache:
    device = model.model.embed_tokens.weight.device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    cache = DynamicCache()

    with torch.no_grad():
        _ = model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True
        )
    return cache

def clean_up(cache: DynamicCache, origin_len: int):
    for i in range(len(cache.key_cache)):
        cache.key_cache[i] = cache.key_cache[i][:, :, :origin_len, :]
        cache.value_cache[i] = cache.value_cache[i][:, :, :origin_len, :]

def get_device_map() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device_map() #cpu

model_name = "meta-llama/Llama-3.2-3B-Instruct"
#model_name= "meta-llama/Llama-3.2-1B"
#model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, token="", trust_remote_code=True) #INSERT HF TOKEN IN ""
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map=device,
    trust_remote_code=True,
    token="" #INSERT HF TOKEN IN ""
)
#disk_offload(model=model, offload_dir="offload")
#device = "cuda" if torch.cuda.is_available() else "cpu"
#model.to(device)
print(f"Loaded {model_name}.")

stime = time.time()
with open("Jon.txt", "r", encoding="utf-8") as f:
    doc_text = f.read()

system_prompt = f"""
<|system|>
You are an assistant who provides concise factual answers. You will read the text file
and respond with a max of 3 sentences. If you do not know, say "I don't know".
<|user|>
Context:
{doc_text}
Question:
""".strip()

ronan_cache = get_kv_cache(model, tokenizer, system_prompt)
origin_len = ronan_cache.key_cache[0].shape[-2]
print("KV cache built.")
etime = time.time()
ttime = etime - stime
print(f"total time: {ttime:.4f} seconds")

stime = time.time()
question1 = "What does Jonathan like?" #Change question to user input
clean_up(ronan_cache, origin_len)
input_ids_q1 = tokenizer(question1 + "\n", return_tensors="pt").input_ids.to(device)
gen_ids_q1 = generate(model, input_ids_q1, ronan_cache)
answer1 = tokenizer.decode(gen_ids_q1[0], skip_special_tokens=True)
print("Q1:", question1)
print("A1:", answer1)

etime = time.time()
ttime = etime - stime
print(f"total time: {ttime:.4f} seconds")

stime = time.time()
question2 = "What abilities can nidorina have?" #Change question to user input
clean_up(ronan_cache, origin_len)
input_ids_q2 = tokenizer(question2 + "\n", return_tensors="pt").input_ids.to(device)
gen_ids_q2 = generate(model, input_ids_q2, ronan_cache)
answer2 = tokenizer.decode(gen_ids_q2[0], skip_special_tokens=True)
print("Q2:", question2)
print("A2:", answer2)

etime = time.time()
ttime = etime - stime
print(f"total time: {ttime:.4f} seconds")

stime = time.time()
question3 = "Give me the hp, def, and atk of meowth, respectively." #Change question to user input
clean_up(ronan_cache, origin_len)
input_ids_q3 = tokenizer(question3 + "\n", return_tensors="pt").input_ids.to(device)
gen_ids_q3 = generate(model, input_ids_q3, ronan_cache)
answer3 = tokenizer.decode(gen_ids_q3[0], skip_special_tokens=True)
print("Q3:", question3)
print("A3:", answer3)

etime = time.time()
ttime = etime - stime
print(f"total time: {ttime:.4f} seconds")

stime = time.time()
question4 = "What is the name of index number 23?" #Change question to user input
clean_up(ronan_cache, origin_len)
input_ids_q4 = tokenizer(question4 + "\n", return_tensors="pt").input_ids.to(device)
gen_ids_q4 = generate(model, input_ids_q4, ronan_cache)
answer4 = tokenizer.decode(gen_ids_q4[0], skip_special_tokens=True)
print("Q4:", question4)
print("A4:", answer4)

etime = time.time()
ttime = etime - stime
print(f"total time: {ttime:.4f} seconds")