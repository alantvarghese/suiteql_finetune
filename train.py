import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = os.getenv("BASE_MODEL", "unsloth/Llama-3.2-3B-Instruct-bnb-4bit")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs/netsuite-sql-lora")

TRAIN_FILE = os.getenv("TRAIN_FILE", "./data/train.jsonl")
VAL_FILE = os.getenv("VAL_FILE", "./data/val.jsonl")

# -----------------------------
# Load model + tokenizer
# -----------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,          # auto-detect
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# -----------------------------
# Load dataset
# -----------------------------
dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VAL_FILE,
    }
)

def format_chat_examples(batch):
    texts = []
    for messages in batch["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(format_chat_examples, batched=True)

# -----------------------------
# Training args
# -----------------------------
bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=20,
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_steps=50,
    save_steps=50,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_total_limit=2,
    bf16=bf16_supported,
    fp16=not bf16_supported,
    optim="adamw_8bit",
    report_to="none",
    seed=3407,
)

# -----------------------------
# Trainer
# -----------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,
    args=training_args,
)

trainer.train()

# Save adapter
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Saved LoRA adapter to: {OUTPUT_DIR}")
