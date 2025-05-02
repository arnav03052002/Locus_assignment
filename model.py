import os
import fitz  
import faiss
import numpy as np
import json
from transformers import AutoTokenizer, AutoModel
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter

# === Environment fix ===
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === Load PDF ===
def load_pdf(path):
    text = ""
    with fitz.open(path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

pdf_path = "Locus Platform Survey Implementation Guide 1.pdf"
text = load_pdf(pdf_path)

# === Chunking ===
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
chunks = splitter.split_text(text)

# === Load embedding model (CPU or GPU) ===
model_name = "BAAI/bge-base-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# === Embed in batches ===
def get_embeddings(texts, batch_size=8):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch = ["Represent this sentence for retrieval: " + t for t in batch]
        tokens = tokenizer(batch, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**tokens)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.extend(embeddings.cpu().numpy())
    return all_embeddings

# === Generate embeddings ===
print("Generating embeddings...")
embeddings = get_embeddings(chunks)
embedding_dim = embeddings[0].shape[0]

# === Create FAISS index ===
print("Creating FAISS index...")
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(embeddings).astype("float32"))

# === Save index + chunks ===
faiss.write_index(index, "vector.index")
with open("chunks.json", "w") as f:
    json.dump(chunks, f)

print("Done! Saved vector.index and chunks.json")

