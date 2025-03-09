from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY! Please set it in the .env file.")

# Initialize Flask app and API
app = Flask(__name__)
api = Api(app)

# Load Data from URL
try:
    loader = WebBaseLoader(["https://brainlox.com/courses/category/technical"])
    documents = loader.load()
except Exception as e:
    raise RuntimeError(f"Error loading webpage: {str(e)}")

# Preprocess Data
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.split_documents(documents)

# Generate Embeddings & Store in Vector Database
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_db = FAISS.from_documents(split_docs, embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# Setup Chat Model and QA Chain
chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY)

# Initialize memory for conversation tracking
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define a custom prompt for better responses
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=
    """
    Use the pieces of information provided in the context to answer user's question.
    If you dont know the answer, just say that you dont know, dont try to make up an answer. 
    Dont provide anything out of the given context
    Question: {question} 
    Context: {context} 
    Answer:
    """
)

# Use RetrievalQA with the custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt},
    memory=memory
)

# Define API Endpoints
class Home(Resource):
    def get(self):
        return {"message": "Welcome to the LangChain Chatbot"}

class Chat(Resource):
    def get(self):
        query = request.args.get("message", "").strip()
        if not query:
            return {"error": "Please provide a message parameter in the URL"}, 400

        try:
            response = qa_chain.run({"query": query}).strip()
            return {"response": response}
        except Exception as e:
            return {"error": str(e)}, 500

    def post(self):
        data = request.get_json()
        query = data.get("message", "").strip()

        if not query:
            return {"error": "Message is required"}, 400

        try:
            response = qa_chain.run({"query": query}).strip()
            return {"response": response}
        except Exception as e:
            return {"error": str(e)}, 500

# Add routes to API
api.add_resource(Home, "/")
api.add_resource(Chat, "/chat")

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
