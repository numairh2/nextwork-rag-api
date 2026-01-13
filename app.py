import uuid
import os
from fastapi import FastAPI
import chromadb
import ollama

app = FastAPI()
client = chromadb.PersistentClient(path="./db")
collection = client.get_or_create_collection("docs")
ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_client = ollama.Client(host=ollama_host)

@app.post("/query")
def query(q: str): 
    results = collection.query(query_texts=[q], n_results=1)
    context = results["documents"][0][0] if results["documents"] else ""

    answer = ollama_client.generate(
        model= "tinyllama",
        prompt=f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer clearly and concisely:"

    )
    return {"answer": answer["response"]}


# How does this RAG FLOW work: Get quetsion => Searches knowlege base with Chroma => Get releveant context => Combine context + question => Sends to Ollamam's tinyllamam and retuns an AI generated answer based on your docs


@app.post("/add")
def add(text: str):
    collection.add(documents=[text], ids=[str(uuid.uuid4())])
    return {"status": "added"}