import csv
import json

# --- KONFIGURASI NAMA FILE ---
INPUT_CSV = 'files_exploits.csv'       # Nama file CSV Exploit-DB Anda
OUTPUT_JSONL = 'dataset_siap_train.jsonl' # Nama file hasil yang akan di-upload ke VPS

print(f"--- Memulai konversi dari {INPUT_CSV} ke {OUTPUT_JSONL} ---")

dataset = []
count_total = 0
count_metasploit = 0

try:
    # Membuka file CSV dengan encoding utf-8 (mengabaikan error karakter aneh)
    with open(INPUT_CSV, mode='r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # 1. FILTERING: Hapus data yang tidak relevan
            # Kita hanya ambil tipe 'remote', 'webapps', dan 'local'. Buang 'dos' (Denial of Service).
            if row['type'] not in ['remote', 'webapps', 'local']:
                continue

            # Ambil data dari kolom
            exploit_id = row['id']
            file_path = row['file']
            description = row['description']
            platform = row['platform']
            exploit_type = row['type']
            
            # Bersihkan deskripsi yang terlalu pendek atau error
            if len(description) < 5 or "OSVDB" in description:
                continue

            # 2. LOGIKA DETEKSI METASPLOIT
            # Cek apakah file berakhiran .rb (Ruby) atau deskripsi menyebut Metasploit
            is_metasploit = False
            tool_name = "Exploit Code"
            
            if file_path.endswith('.rb') or "Metasploit" in description:
                is_metasploit = True
                count_metasploit += 1
                tool_name = "Metasploit Module"

            # 3. MEMBUAT FORMAT PERCAKAPAN (INSTRUCTION TUNING)
            
            # Instruksi: Seolah-olah user meminta analisa keamanan
            instruction_text = (
                f"Analisa sistem dengan platform '{platform}'. "
                f"Terdeteksi potensi kerentanan: {description}. "
                f"Bagaimana identifikasi dan referensinya?"
            )

            # Respon AI: Memberikan informasi teknis
            response_text = (
                f"Kerentanan ini teridentifikasi sebagai serangan tipe **{exploit_type}**.\n"
                f"Tercatat di database Exploit-DB dengan ID: {exploit_id}.\n\n"
            )

            if is_metasploit:
                response_text += (
                    f"**SOLUSI PENGUJIAN:**\n"
                    f"Kerentanan ini memiliki modul **Metasploit Framework** yang valid.\n"
                    f"Path Modul: `{file_path}`\n"
                    f"Gunakan modul tersebut untuk memverifikasi keamanan sistem."
                )
            else:
                response_text += (
                    f"**REFERENSI KODE:**\n"
                    f"Terdapat script Proof-of-Concept (PoC) manual untuk celah ini.\n"
                    f"Lokasi file: `{file_path}`"
                )

            # Masukkan ke list dataset
            entry = {
                "text": f"### Instruction: {instruction_text}\n\n### Response: {response_text}"
            }
            dataset.append(entry)
            count_total += 1

    # Simpan hasil ke file JSONL
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')

    print("--- KONVERSI SELESAI! ---")
    print(f"Total Data Valid: {count_total} baris")
    print(f"Modul Metasploit Ditemukan: {count_metasploit}")
    print(f"File siap upload: {OUTPUT_JSONL}")

except FileNotFoundError:
    print("ERROR: File 'files_exploits.csv' tidak ditemukan.")
    print("Pastikan file CSV berada di folder yang sama dengan script ini.")
except Exception as e:
    print(f"Terjadi error: {e}")
