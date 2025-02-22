from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
import anthropic
import os
import json
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_API_KEY_HERE")

def build_database():
    if os.path.exists("eaton_db") and os.path.exists("eaton_fire_docs.json"):
        logger.info("Database already exists, skipping initialization.")
        return

    logger.info("Building database on startup...")
    urls = [
        "https://laist.com/brief/news/climate-environment/researchers-tested-sandboxes-street-dust-lead-eaton-fire",
        "https://www.kcrw.com/culture/shows/good-food/fire-soil-safety-lunar-new-year-china-dishes/eaton-palisades-fire-soil-ash-residue-fallout-danger-garden-fruit-vegetables",
        "https://abc7.com/post/toxic-dangers-linger-inside-altadena-homes-survived-eaton-fire/15896312/"
        # Reduced to 3 URLs for faster startup; add more later if needed
    ]

    documents = []
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text(separator=" ")
            documents.append({"url": url, "content": text})
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for doc in documents:
        split_texts = splitter.split_text(doc["content"])
        for i, text in enumerate(split_texts):
            chunks.append({"url": doc["url"], "text": text, "id": f"{doc['url']}_{i}"})

    with open("eaton_fire_docs.json", "w") as f:
        json.dump(chunks, f)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="./eaton_db")
    collection = client.get_or_create_collection("eaton_fire_docs")
    embeddings_list = [embeddings.embed_query(chunk["text"]) for chunk in chunks]
    collection.add(
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[{"url": chunk["url"]} for chunk in chunks],
        ids=[chunk["id"] for chunk in chunks],
        embeddings=embeddings_list
    )
    logger.info("Database initialized on startup.")

# Initialize embeddings and Chroma
build_database()
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    collection_name="eaton_fire_docs",
    embedding_function=embeddings,
    persist_directory="./eaton_db"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Using this info: {context}\nAnswer this question naturally: {question}\nInclude citations (e.g., 'Source: [URL]') and end with: 'Note: I’m not a doctor or substitute for professional advice—contact an expert for definitive answers.'"
)

def answer_question(question):
    try:
        docs = retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in docs])
        sources = "\n".join([f"Source: {doc.metadata['url']}" for doc in docs])
        prompt_text = prompt.format(context=context, question=question)
        
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt_text}]
        )
        answer = response.content[0].text
        return f"{answer}\n\n{sources}\nNote: I’m not a doctor or substitute for professional advice—contact an expert for definitive answers."
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return f"Error: {str(e)}"

@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    answer = answer_question(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
