from sentence_transformers import SentenceTransformer
import chromadb
import json

with open("eaton_fire_docs.json", "r") as f:
    chunks = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./eaton_db")
collection = client.get_or_create_collection("eaton_fire_docs")

embeddings = [model.encode(chunk["text"]).tolist() for chunk in chunks]
collection.add(
    documents=[chunk["text"] for chunk in chunks],
    metadatas=[{"url": chunk["url"]} for chunk in chunks],
    ids=[chunk["id"] for chunk in chunks],
    embeddings=embeddings
)

print("Vector database built and saved in ./eaton_db")
