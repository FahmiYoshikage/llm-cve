import torch
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# --- KONFIGURASI ---
MODEL_ID = "deepseek-ai/deepseek-coder-33b-instruct"
OUTPUT_DIR = "./hasil_training_vuln"
DATASET_FILE = "dataset_siap_train.jsonl" 

# Setup device
device_map = "auto"

print(f"--- Memulai Setup untuk Model: {MODEL_ID} ---")

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" 

# 2. Load Base Model (Full Precision bfloat16 untuk MI300X)
print("--- Loading Model ke VRAM (bfloat16) ---")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
    trust_remote_code=True,
    use_cache=False
)

# 3. Konfigurasi LoRA
peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# 4. Load Dataset
print(f"--- Loading Dataset: {DATASET_FILE} ---")
dataset = load_dataset("json", data_files=DATASET_FILE, split="train")

# 5. Konfigurasi Training (HYBRID FIX)
# dataset_text_field kita taruh di sini (Sesuai error pertama)
# max_seq_length kita HAPUS dari sini (Sesuai error kedua)
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",       # OK: Masuk Config
    packing=False,                   # OK: Masuk Config
    num_train_epochs=3,
    per_device_train_batch_size=8,   
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    weight_decay=0.001,
    fp16=False,
    bf16=True,                       
    logging_steps=1,
    save_strategy="epoch",
    report_to="none",
    optim="adamw_torch",
    max_seq_length=2048              # Kita coba paksa di sini dulu karena TRL versi baru maunya di sini
)

# 6. Inisialisasi Trainer
# Jika script ini masih error 'max_seq_length' di SFTConfig, 
# Hapus baris 'max_seq_length' di atas, dan uncomment baris di bawah ini.
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_args,
    # max_seq_length=2048, # Pindahkan ke sini JIKA error lagi
)

# 7. Mulai Training
print("--- Mulai Proses Training ---")
trainer.train()

# 8. Simpan Model
print(f"--- Menyimpan Model ke {OUTPUT_DIR} ---")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("--- SELESAI ---")
