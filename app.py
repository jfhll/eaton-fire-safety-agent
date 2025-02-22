from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
import anthropic
import os

app = Flask(__name__)

# Use environment variable or fallback to placeholder
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_API_KEY_HERE")

# Initialize embeddings with HuggingFace model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Chroma with the embedding function
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
