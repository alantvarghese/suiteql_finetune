import os
from unsloth import FastLanguageModel
from peft import PeftModel

BASE_MODEL = os.getenv("BASE_MODEL", "unsloth/Llama-3.2-3B-Instruct-bnb-4bit")
ADAPTER_DIR = os.getenv("ADAPTER_DIR", "./outputs/netsuite-sql-lora")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048"))

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

model = PeftModel.from_pretrained(model, ADAPTER_DIR)
FastLanguageModel.for_inference(model)

messages = [
    {
        "role": "system",
        "content": "You are a NetSuite SQL assistant. Generate safe SuiteQL only. Return only SQL unless asked to explain."
    },
    {
        "role": "user",
        "content": "Show open AR by customer for last quarter."
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.1,
    do_sample=False,
)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
