import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- KONFIGURASI ---
BASE_MODEL_ID = "deepseek-ai/deepseek-coder-33b-instruct"
ADAPTER_PATH = "./hasil_training_vuln" # Folder hasil training Anda

print(f"--- 1. Loading Base Model: {BASE_MODEL_ID} ---")
# Load model asli (Otak Dasar)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16, # Wajib sama dengan saat training (MI300X)
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

print(f"--- 2. Loading Adapter LoRA: {ADAPTER_PATH} ---")
# Gabungkan dengan hasil training Anda (Ilmu Security)
# AI akan menjadi: DeepSeek Coder + Pengetahuan ExploitDB
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval() # Mode ujian (tidak belajar lagi)

print("\n==================================================")
print("🤖 AI SECURITY ADVISOR SIAP!")
print("Ketik 'exit' untuk keluar.")
print("==================================================\n")

while True:
    # Input user interaktif
    try:
        user_input = input("\n[USER] Masukkan request/CVE: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        if not user_input.strip():
            continue

        # Format Prompt (WAJIB SAMA PERSIS dengan saat Training!)
        # Ingat dataset kita: "### Instruction: ... ### Response:"
        prompt = f"### Instruction: {user_input}\n\n### Response:"

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        print("[AI] Sedang mencari di database otak...")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512, # Panjang jawaban maksimal
                temperature=0.6,    # 0.1 = Kaku/Hafalan, 1.0 = Kreatif/Ngawur
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode jawaban komputer ke teks manusia
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Bersihkan output (ambil bagian setelah Response saja)
        if "### Response:" in response:
            final_answer = response.split("### Response:")[1].strip()
        else:
            final_answer = response

        print(f"\n[AI ANSWER]:\n{final_answer}\n")
        print("-" * 50)
        
    except KeyboardInterrupt:
        print("\nKeluar...")
        break
    except Exception as e:
        print(f"Error: {e}")
