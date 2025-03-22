import os
import json
import numpy as np
from dotenv import load_dotenv
import faiss
import torch
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL = os.getenv("MODEL")
JSON_DIR = os.getenv("JSON_DIR")
FAISS_INDEX = os.getenv("FAISS_INDEX")
JSON_INDEX = os.getenv("JSON_INDEX")


def load_json_files(directory):
    texts = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    data = json.load(file)
                    title = data["title"]
                    content = data["content"]
                    texts.append("\n".join([title, content]))
                    filenames.append(filename)
                except json.JSONDecodeError as e:
                    print(f"Error decoding {filename}: {e}")
    return texts, filenames


device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL, cache_folder=MODEL, device=device)

texts, filenames = load_json_files(JSON_DIR)
embeddings = np.array([model.encode(text) for text in texts])

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, FAISS_INDEX)
np.save(JSON_INDEX, np.array(filenames))

print("Embeddings and FAISS index saved successfully!")
