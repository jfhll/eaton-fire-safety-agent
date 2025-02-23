import os
import logging
from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
import anthropic
import json
import chromadb
from langchain_text_splitter import RecursiveCharacterTextSplitter

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_API_KEY_HERE")

# Global variables (loaded on startup)
embeddings = None
vectorstore = None
retriever = None
db_initialized = False

def create_or_load_vectorstore(chunks):
    global embeddings, vectorstore, retriever, db_initialized
    try:
        logger.info("Creating or loading vector store...")
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create or load Chroma vector store
        if not os.path.exists("eaton_db"):
            os.makedirs("eaton_db", exist_ok=True)
        
        client = chromadb.PersistentClient(path="./eaton_db")
        collection = client.get_or_create_collection("eaton_fire_docs")
        
        # If the collection is empty, add the pre-processed data
        if len(collection.count()) == 0:
            embeddings_list = [embeddings.embed_query(chunk["text"]) for chunk in chunks]
            collection.add(
                documents=[chunk["text"] for chunk in chunks],
                metadatas=[{"url": chunk["url"]} for chunk in chunks],
                ids=[chunk["id"] for chunk in chunks],
                embeddings=embeddings_list
            )
        
        # Initialize vector store and retriever
        vectorstore = Chroma(
            collection_name="eaton_fire_docs",
            embedding_function=embeddings,
            persist_directory="./eaton_db"
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})  # Fewer documents for speed
        db_initialized = True
        logger.info("Vector store created or loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to create or load vector store: {e}")
        db_initialized = False

def load_preprocessed_data():
    global embeddings, vectorstore, retriever, db_initialized
    try:
        logger.info("Loading pre-processed data...")

        # Load pre-processed chunks from JSON
        if not os.path.exists("eaton_fire_docs.json"):
            raise Exception("eaton_fire_docs.json not found locally")
        
        with open("eaton_fire_docs.json", "r") as f:
            chunks = json.load(f)
        
        create_or_load_vectorstore(chunks)
    except Exception as e:
        logger.error(f"Failed to load pre-processed data: {e}")
        db_initialized = False

# Load pre-processed data on startup
load_preprocessed_data()

# Health check route
@app.route("/health", methods=["GET"])
def health():
    logger.info("Health check requested")
    return jsonify({"status": "ok"})

# Status check route
@app.route("/status", methods=["GET"])
def status():
    return jsonify({"db_initialized": db_initialized})

# API ask route
@app.route("/api/ask", methods=["POST"])
def api_ask():
    global db_initialized, retriever
    logger.info("Received request to /api/ask")
    data = request.get_json()
    if data is None:
        logger.info("Invalid JSON")
        return jsonify({"error": "Invalid JSON"}), 400
    question = data.get("question", "")
    if not question:
        logger.info("No question provided")
        return jsonify({"error": "No question provided"}), 400
    if not db_initialized:
        logger.info("Database not yet initialized")
        return jsonify({"error": "Database not yet initialized, please try again in a moment"}), 503
    try:
        logger.info("Retrieving documents")
        docs = retriever.get_relevant_documents(question)
        logger.info("Documents retrieved")
        context = "\n".join([doc.page_content for doc in docs])
        sources = "\n".join([f"Source: {doc.metadata['url']}" for doc in docs])
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="Using this info: {context}\nAnswer this question naturally: {question}\nInclude citations (e.g., 'Source: [URL]') and end with: 'Note: I'm not a doctor or substitute for professional advice—contact an expert for definitive answers.'"
        )
        prompt_text = prompt.format(context=context, question=question)
        
        logger.info("Calling Anthropic API")
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=300,  # Reduced for speed
            messages=[{"role": "user", "content": prompt_text}]
        )
        logger.info("Anthropic API call successful")
        answer = response.content[0].text
        return jsonify({"answer": f"{answer}\n\n{sources}\nNote: I'm not a doctor or substitute for professional advice—contact an expert for definitive answers."})
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return jsonify({"answer": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render default
    logger.info(f"Starting Flask on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
