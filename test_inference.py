import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "deepseek-ai/deepseek-coder-33b-instruct"
ADAPTER_DIR = "./hasil_training_vuln" # Folder hasil training tadi

print("--- Loading Base Model & Adapter ---")
# 1. Load Base
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# 2. Gabungkan dengan Adapter hasil training Anda
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

# 3. Test Prompt
input_text = """### Instruction: Cek keamanan kode berikut.
void login(char *pass) {
    char buff[10];
    strcpy(buff, pass);
}

### Response:"""

inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

print("--- Generating Response ---")
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=200, 
        temperature=0.7,
        do_sample=True
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
