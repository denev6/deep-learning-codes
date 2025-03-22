import torch
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS", device=device)
model.save("snunlp/KR-SBERT-V40K-klueNLI-augSTS")