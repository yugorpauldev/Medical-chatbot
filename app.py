from flask import Flask, render_template, jsonify, request
from src.helper import download_embedding
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

#Add environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY


#Add Embedding model

embedding = download_embedding()

index_name = "medical-chatbot"

# Initialize the Pinecone store
docsearch = PineconeVectorStore(
    embedding=embedding,
    index_name=index_name
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


#initializing the model
chatModel = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2
)

# Create a query reformulation chain with subject matter analysis
query_reformulation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an AI assistant specialized in medical terminology and subject matter analysis.

TASK: Analyze the user's question to determine the medical specialty and reformulate it for better retrieval.

SUBJECT ANALYSIS:
- If the question relates to blood, blood cells, hemoglobin, anemia, blood disorders, hematology tests, or blood-related pathology, focus on HEMATOLOGY terms
- If the question relates to organ systems, body functions, anatomy, physiology, or general body processes, focus on PHYSIOLOGY terms
- If the question spans both areas, include terms from both specialties

REFORMULATION GUIDELINES:
- Expand medical abbreviations (e.g., Hb → hemoglobin, RBC → red blood cells)
- Include relevant medical terminology and synonyms
- Make the query more specific and technical
- Add context words that would appear in medical textbooks
- Do not answer the question, only reformulate it

Return only the reformulated query without explanations."""),
        ("human", "{input}")
    ]
)

query_reformulation_chain = query_reformulation_prompt | chatModel | (lambda x: x.content)

# Create the RAG prompt using the system prompt from prompt.py
rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# Create the question-answer chain
question_answer_chain = create_stuff_documents_chain(chatModel, rag_prompt)


def reformulate_query(inputs):
    # Get the original query
    original_query = inputs["input"]
    
    # Reformulate the query - the result is the content string only
    reformulated_query = query_reformulation_chain.invoke({"input": original_query})
    
    # Log the reformulated query for debugging
    print(f"Original query: {original_query}")
    print(f"Reformulated query: {reformulated_query}")
    
    # Return both queries
    return {
        "original_input": original_query,
        "input": reformulated_query
    }

# The complete RAG chain with query reformulation
rag_chain = (
    RunnablePassthrough.assign(
        reformulated_input = lambda x: reformulate_query(x)
    )
    | {"input": lambda x: x["reformulated_input"]["input"], 
       "original_input": lambda x: x["reformulated_input"]["original_input"]}
    | create_retrieval_chain(retriever, question_answer_chain)
)



@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print("Original user query:", input)
    
    # Invoke the RAG chain with query reformulationccccv
    response = rag_chain.invoke({"input": msg})
    
    # Let the LLM handle citations naturally - no additional source formatting
    print("Response:", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)