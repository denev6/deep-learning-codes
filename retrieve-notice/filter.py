import os
import json
from dotenv import load_dotenv

load_dotenv()

JSON_DIR = os.getenv("JSON_DIR")
MIN_LENGTH = 5

total = 0
failed = 0

for filename in os.listdir(JSON_DIR):
    if filename.endswith(".json"):
        file_path = os.path.join(JSON_DIR, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        total += 1
        if len(data["content"]) < MIN_LENGTH:
            os.remove(file_path)
            failed += 1
            print(f"{file_path} removed.")

print(f"\n{failed} files failed.")
print(f"\n{total - failed} files remaining.")
