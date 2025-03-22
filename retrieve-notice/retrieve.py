import os
import faiss
import numpy as np
import json
import torch
import time
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"
FAISS_INDEX = os.getenv("FAISS_INDEX")
JSON_INDEX = os.getenv("JSON_INDEX")
MODEL = os.getenv("MODEL")
DATA_DIR = os.getenv("JSON_DIR")
BASE_URL = os.getenv("BASE_URL")

index = faiss.read_index(FAISS_INDEX)
filenames = np.load(JSON_INDEX)
model = SentenceTransformer(MODEL, cache_folder=MODEL, device=device)

def load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    return data["title"], data["content"]

def retrieve(query, k=3):
    query_embedding = model.encode(query, device=device).reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    retrieved_files = [filenames[i] for i in indices[0]]
    return retrieved_files, distances

def iter_retrieve_for_testing(query, k=3):
    yield f"\n[{str(device).upper()}] 검색 중...\n"
    start = time.time()
    retrieved_files, _ = retrieve(query, k)
    for idx, filename in enumerate(retrieved_files, start=1):
        title, content = load_json(filename)
        yield f"--- [{idx}] {BASE_URL}?mode=view&articleNo={filename.rsplit('.', 1)[0]} ---\n"
        yield f"Title: {title}\n"
        yield f"Content: {content}\n"

    end = time.time()
    yield f"\n검색 완료: {end - start:.3f}초 소요"


if __name__ == "__main__":
    query = "마이크로디그리"
    for info in iter_retrieve_for_testing(query):
        print(info)
