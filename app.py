import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from flask import Flask, request, jsonify

from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY'] # using GROQ to run open source models

# loading the website using WebBaseLoader
url = "https://brainlox.com/courses/category/technical"
loader = WebBaseLoader(url)
document = loader.load()

# splitting the text 
text_splitter = RecursiveCharacterTextSplitter()
final_document = text_splitter.split_documents(document)

# creating embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector =  FAISS.from_documents(final_document, embeddings)

# prompt and llm model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
prompt_template = ChatPromptTemplate.from_template(
        """
        Answer every question only from the provided context.
        Please provide the most accurate response based on the question
        <context>
        {context}
        </context>
        Question: {input}
        """
    )

# chain
chain = create_stuff_documents_chain(llm, prompt_template)
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, chain)

# flask app
app = Flask(__name__)
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        question = data.get('question')
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        # Pass the question to the retrieval chain
        response = retrieval_chain({'input': question})
        return jsonify({'response': response['output_text']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)










