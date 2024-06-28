from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os

# Load environment variables securely
load_dotenv()

# Function to validate and set Google API key
def validate_api_key():
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("Missing GOOGLE_API_KEY environment variable. Please set it before running the application.")

# URL to be loaded
url = "https://brainlox.com/courses/category/technical"

# Create a Flask app
app = Flask(__name__)

# Variable to store conversation history
conversation_history = []

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"message": "Chatbot is up and running!"})

@app.route("/ask", methods=["POST"])
def answer_question():
    validate_api_key()

    user_question = request.json.get("question")
    if not user_question:
        return jsonify({"error": "Missing question in request body."}), 400

    # Create a WebBaseLoader instance
    loader = WebBaseLoader(url)

    # Load the documents
    documents = loader.load()

    # Function to create text chunks from documents
    def get_text_chunks(documents):
        text_chunks = []
        for doc in documents:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
            chunks = text_splitter.split_text(doc.page_content)
            text_chunks.extend(chunks)
        return text_chunks

    text_chunks = get_text_chunks(documents)

    # Create vector store if it doesn't exist
    vector_store_path = "faiss_index"
    if not os.path.exists(vector_store_path):
        # Generate embeddings using Google Generative AI Embeddings
        embeddings = GoogleGenerativeAIEmbeddings().encode(text_chunks)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(vector_store_path)
    else:
        vector_store = FAISS.load_local(vector_store_path)

    # Update conversation history with current question
    conversation_history.append(user_question)

    # Create a RetrievalQAChain with message history logic
    chain = LLMChain(
        llm="gemini-pro",  # Use Gemini Pro model
        prompt_template=ChatPromptTemplate.from_template(
            "Here's what I found relevant to your previous questions: {history}\nNow, you asked: {question}\nAnswer: {answer}"
        ),
        retriever=vector_store.as_retriever(),  # Retrieve relevant documents from vector store
    )

    # Answer the user's question
    answer = chain.run({"history": conversation_history[-2:], "question": user_question})

    return jsonify({"answer": answer})

if __name__ == "__main__":
    validate_api_key()  # Ensure API key is set before running
    app.run(debug=True)
