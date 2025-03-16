import os
import csv
import requests

script_dir = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = os.path.join(script_dir, "dataset")  
DATA_DIR = os.path.join(BASE_DIR, "data")  
ERROR_FILE = os.path.join(BASE_DIR, "eroare.txt")  
CSV_FILE = os.path.join(script_dir, "logos.csv")  

os.makedirs(DATA_DIR, exist_ok=True)

with open(CSV_FILE, newline='', encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)  
    domains = sorted(set(row[0].strip().lower() for row in reader if row))  

def download_logo(domain):
    logo_url = f"https://www.google.com/s2/favicons?domain={domain}&sz=256"
    output_file = os.path.join(DATA_DIR, f"{domain}.png")

    try:
        response = requests.get(logo_url, timeout=5)
        if response.status_code == 200 and response.content:
            with open(output_file, "wb") as f:
                f.write(response.content)
            print(f"Logo salvat: {domain}")
        else:
            raise Exception("Logo indisponibil")
    except Exception:
        with open(ERROR_FILE, "a", encoding="utf-8") as err_file:
            err_file.write(domain + "\n")
        print(f"Eroare pentru: {domain}")

for domain in domains:
    download_logo(domain)

print("\nProces finalizat!")
