from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
import google.generativeai as genai
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

# Load API keys from environment variables
PINECONE_API_KEY = os.getenv("pcsk_243Jdr_H6mpzL6nHphJhSeR8qn6oza5WWV5m4UCykofoAeSWnREYjR3fepZRDCBNaJfmEN")
GEMINI_API_KEY = os.getenv("AIzaSyCR2GGbvlTz6EOgNmYH4uA4aW1_R3vWzmk")  # Use the correct key for Gemini

# Set the API key for Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Load embeddings for document search
embeddings = download_hugging_face_embeddings()

index_name = "lla-chatbot"

# Embed each chunk and upsert the embeddings into your Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define Gemini 1.5 Flash model
def call_gemini(input_text):
    model = genai.GenerativeModel("gemini-1.5-flash")  # Use Gemini 1.5 Flash
    response = model.generate_content(input_text)
    return response.text if response.text else "Sorry, I couldn't generate a response."

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create a custom chain using Gemini
def question_answer_chain(input_text):
    return call_gemini(input_text)

rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User Input:", msg)
    response = call_gemini(msg)
    print("Response:", response)
    return str(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
