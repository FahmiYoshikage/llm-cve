import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,  # Pakai ini, bukan SFTConfig
)
from peft import LoraConfig
from trl import SFTTrainer # Hapus SFTConfig dari sini

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

# 2. Load Base Model
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

# 5. Konfigurasi Training (GAYA LAMA - STABIL)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    
    # --- UBAHAN PENTING ---
    per_device_train_batch_size=4,   # Turunkan dari 8 ke 4 (biar gak crash memori)
    gradient_accumulation_steps=4,   # Naikkan dari 2 ke 4 (biar hasil tetap bagus)
    
    save_strategy="steps",           # Ubah dari "epoch" ke "steps"
    save_steps=50,                   # Nyimpen setiap 50 langkah (sering!)
    save_total_limit=2,              # Cuma simpan 2 file save terakhir (biar disk gak penuh)
    # ----------------------
    
    learning_rate=2e-5,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    logging_steps=1,
    report_to="none",
    optim="adamw_torch",
)
# 6. Inisialisasi Trainer
# Perhatikan: max_seq_length dan dataset_text_field masuk SINI (bukan di args)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_args,
    dataset_text_field="text",  # INI CARA VERSI 0.8.6
    max_seq_length=2048,        # INI CARA VERSI 0.8.6
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
