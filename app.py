import os
import logging
from flask import Flask, request, jsonify
import anthropic
import json
import chromadb
from sentence_transformers import SentenceTransformer
import time
from functools import wraps

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
RATE_LIMIT = 30  # requests per minute
RATE_WINDOW = 60  # seconds

# Global state
db_client = None
collection = None
embedder = None
request_times = []

def rate_limit_check():
    """Simple rolling window rate limiter"""
    now = time.time()
    global request_times
    request_times = [t for t in request_times if now - t < RATE_WINDOW]
    if len(request_times) >= RATE_LIMIT:
        return False
    request_times.append(now)
    return True

def rate_limited(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not rate_limit_check():
            return jsonify({"error": "Rate limit exceeded"}), 429
        return f(*args, **kwargs)
    return decorated_function

def init_database():
    """Initialize the vector database and embedder"""
    global db_client, collection, embedder
    
    try:
        logger.info("Initializing database and embedder...")
        
        # Initialize sentence transformer
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        db_client = chromadb.PersistentClient(path="./eaton_db")
        
        # Get or create collection
        collection = db_client.get_or_create_collection(
            name="eaton_fire_docs",
            metadata={"description": "Eaton fire safety documentation"}
        )
        
        # Load initial data if collection is empty
        if collection.count() == 0:
            logger.info("Loading initial data...")
            with open("eaton_fire_docs.json", "r") as f:
                docs = json.load(f)
                
            texts = [doc["text"] for doc in docs]
            metadata = [{"url": doc["url"]} for doc in docs]
            ids = [doc["id"] for doc in docs]
            
            # Generate embeddings
            embeddings = embedder.encode(texts).tolist()
            
            # Add to collection
            collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadata,
                ids=ids
            )
            
        logger.info("Database initialization complete")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False

@app.route('/health')
@rate_limited
def health_check():
    """Basic health check endpoint"""
    return jsonify({
        "status": "healthy",
        "database": db_client is not None and collection is not None,
        "embedder": embedder is not None
    })

@app.route('/api/ask', methods=['POST'])
@rate_limited
def ask():
    """Main query endpoint"""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    try:
        data = request.get_json()
        question = data.get('question')
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
            
        # Generate embedding for question
        question_embedding = embedder.encode(question).tolist()
        
        # Query collection
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=2  # Get top 2 most relevant chunks
        )
        
        # Prepare context from results
        contexts = results['documents'][0]
        sources = [m['url'] for m in results['metadatas'][0]]
        
        # Prepare prompt
        context_text = "\n".join(contexts)
        prompt = f"""You are a helpful AI assistant providing information about the Eaton fire. 
Using only the following information, answer the question naturally and conversationally. 
If you're not sure about something, say so.

Information:
{context_text}

Question: {question}

Remember to cite your sources using the format (Source: URL) after relevant statements."""

        # Get response from Claude
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        answer = response.content[0].text
        
        # Add sources
        source_text = "\n\nSources:\n" + "\n".join(f"- {url}" for url in sources)
        
        return jsonify({
            "answer": answer + source_text
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Initialize database on startup
    if not init_database():
        logger.error("Failed to initialize database. Exiting.")
        exit(1)
        
    # Start server
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
