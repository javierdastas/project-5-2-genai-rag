from flask import Flask, render_template, request, jsonify
import os
import openai
from dotenv import load_dotenv

from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

# OpenAI Approach
from langchain_openai import OpenAIEmbeddings

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'secret-key'

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
gpt_model = "gpt-4o-mini"
openai.api_key = openai_api_key

# Initialize embedding model
open_ai_embedding_model = OpenAIEmbeddings(model='text-embedding-ada-002', api_key=openai_api_key)

# Load the persisted ChromaDB
vector_db = Chroma(persist_directory="/srv/dataset/chroma_db_openai", embedding_function=open_ai_embedding_model)

docs_count = vector_db._collection.count()

# Function to build prompt for OpenAI
def build_prompt(query, context):
    '''
    Arg(s):
        - query (str): User query or question?
        - context (str): top query results from ChrobaDB
    Return:
        - (str): LLM user prompt
    '''        
    return f"""
    ## INTRODUCTION
    You are a Chatbot designed to help answer questions about Puerto Rico.
    The user asked: "{query}"
    
    ## CONTEXT
    News Documentation:
    '''
    {context}
    '''
    
    ## RESTRICTIONS
    Refer to the news by their titles.
    Be clear, transparent, and factual: only state what is in the context without providing opinions or subjectivity.
    Answer the question based solely on the context above; if you do not know the answer, be clear with the user that you do not know.
    Only respond to questions related to the products, avoiding jokes, offensive remarks, and discussions on religion or sexuality.
    If the user does not provide sufficient context, do not answer and instead ask for more information on what the user wants to know.
    
    ## TASK
    First, answer directly to the user, if possible.
    Second, point the user in the right direction of the documentation.
    Lastly, please make sure to respond in JSON structure format only, no markdown or additional formatting.
    
    ## RESPONSE STRUCTURE:
    - title: [Answer Title]
    - answer: [answer text]
    - sources: [source title 1, source title 2, ...]
    - metadata-sources: [ metadata source 1, metadata source 2, ... ]
    """

# Function to query OpenAI and get a response
def generate_response_with_openai(query, k=10):
    '''
        Arg(s):
            - query (str): User query or question.
            - k (int): number of vectors to returns from ChromaDB (default value 10)
        Return:
            - (str): LLM result
    '''      
    # Embedd the Query
    query_embedding = open_ai_embedding_model.embed_query(query)
    # query_embedding = embedding_model.embed_query(query)

    # Search the Top 10 relevant chunks in ChromaDB
    results = vector_db.similarity_search(query, k=k)

    # Search the Top 10 relevant chunks in ChromaDB using embedding query for more precission
    # results = vector_db.similarity_search_by_vector(query_embedding, k=k)
    
    # Collect vector IDs, page content, and any metadata if needed
    vector_ids = [result.id for result in results]  # Extract the vector IDs

    # Join the chunks as one context
    context = "\n".join([result.page_content for result in results])
    
    # Build prompt for OpenAI
    prompt = build_prompt(query, context)
    message = [{'role': 'user', 'content': prompt}]
    
    # Generate response
    response = openai.chat.completions.create(
        messages=message,
        model=gpt_model,
        temperature=0.4,
        max_tokens=3000
    )
    
    response_text = response.choices[0].message.content
    # return parsed_response
    return response_text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_query = request.form.get('query')
        if not user_query:
            return jsonify({"error": "No query provided"}), 400

        try:
            # Get response from OpenAI
            response = generate_response_with_openai(user_query)
            return jsonify(response)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
