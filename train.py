import torch
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# --- KONFIGURASI ---
MODEL_ID = "deepseek-ai/deepseek-coder-33b-instruct"
OUTPUT_DIR = "./hasil_training_vuln"
DATASET_FILE = "dataset_siap_train.jsonl"

# Setup device (Pastikan script jalan di GPU)
device_map = "auto" 

print(f"--- Memulai Setup untuk Model: {MODEL_ID} ---")

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Penting untuk fp16/bf16 training

# 2. Load Base Model (Full Precision bfloat16 untuk MI300X)
print("--- Loading Model ke VRAM (bfloat16) ---")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,  # Format native terbaik untuk MI300X
    device_map=device_map,
    trust_remote_code=True,
    use_cache=False # Matikan cache saat training
)

# 3. Konfigurasi LoRA (Fine-Tuning)
# Karena VRAM besar, kita bisa pakai rank (r) yang tinggi agar hasil lebih bagus
peft_config = LoraConfig(
    r=64,               # Rank tinggi = adaptasi lebih detail
    lora_alpha=128,     # Alpha biasanya 2x dari rank
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[    # Target semua linear layer agar model benar-benar paham coding
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

# 4. Load Dataset
dataset = load_dataset("json", data_files=DATASET_FILE, split="train")

# 5. Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,              # Berapa kali muterin dataset
    per_device_train_batch_size=8,   # Batch size. Di MI300X bisa coba naik ke 16 atau 32!
    gradient_accumulation_steps=2,
    learning_rate=2e-5,              # LR standar untuk fine-tuning
    weight_decay=0.001,
    fp16=False,                      # Jangan pakai fp16
    bf16=True,                       # PAKAI bf16 (Wajib untuk MI300X stabilitas)
    logging_steps=1,
    save_strategy="epoch",
    report_to="none",                # Matikan wandb supaya tidak perlu login
    optim="adamw_torch",
)

# 6. Inisialisasi Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=2048,             # Bisa dinaikkan ke 4096 jika input kode panjang
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
)

# 7. Mulai Training
print("--- Mulai Proses Training ---")
trainer.train()

# 8. Simpan Model
print(f"--- Menyimpan Model ke {OUTPUT_DIR} ---")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("--- SELESAI ---")
