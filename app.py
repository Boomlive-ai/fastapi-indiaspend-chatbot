import os, requests
from flask import Flask, request, jsonify
from bot import Chatbot
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from utils import extract_sources_and_result, prioritize_sources
from tools import fetch_questions_on_latest_articles_in_IndiaSpend
from vectorstore import StoreCustomRangeArticles, StoreDailyArticles
from flask_cors import CORS  # Import CORS

# Initialize Flask 

app = Flask(__name__)
CORS(app)  # Allow all origins
mybot=Chatbot()
workflow=mybot()
application = app

# Initialize Environment Variables

load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")


@app.route('/', methods=['GET'])
def documentation():
    """
    Route to provide API documentation.
    """
    docs = {
        "endpoints": [
            {
                "route": "/query",
                "method": "GET",
                "description": "Query the chatbot with a question and thread ID.",
                "parameters": {
                    "question": "The question to ask the chatbot (required).",
                    "thread_id": "The thread ID for context (required)."
                },
                "response": {
                    "response": "The chatbot's response.",
                    "sources": "Sources related to the response."
                }
            },
            {
                "route": "/store_articles",
                "method": "POST",
                "description": "Store articles for a custom date range.",
                "body": {
                    "from_date": "Start date in 'YYYY-MM-DD' format (required).",
                    "to_date": "End date in 'YYYY-MM-DD' format (required)."
                },
                "response": "Status and details of stored articles."
            },
            {
                "route": "/store_daily_articles",
                "method": "POST",
                "description": "Fetch and store daily articles.",
                "response": "Status and details of stored daily articles."
            },
            {
                "route": "/generate_questions",
                "method": "GET",
                "description": "Generate questions from the latest articles in IndiaSpend.",
                "response": "Generated questions and their details."
            },
            {
                "route": "/documentation",
                "method": "GET",
                "description": "Get the API documentation for available endpoints."
            }
        ]
    }
    return jsonify(docs), 200


@app.route('/query', methods=['GET'])
def query_bot():
    question = request.args.get('question')
    thread_id = request.args.get('thread_id')
    sources = []
    if not question or not thread_id:
        return jsonify({"error": "Missing required parameters"}), 400

    input_data = {"messages": [HumanMessage(content=question)]}
    try:
        response = workflow.invoke(input_data, config={"configurable": {"thread_id": thread_id}})
        result = response['messages'][-1].content
        # print("response['messages'][-1]",response)
        result,raw_sources  = extract_sources_and_result(result)
        sources = prioritize_sources(result, raw_sources)
        if not result:
            result = "No response generated. Please try again."
        return jsonify({"response": result, "sources": sources})
    except Exception as e:
        print(f"Error in query_bot: {str(e)}")
        return jsonify({"error": str(e)}), 500




# Route for Storing Articles with Custom Date Range
@app.route('/store_articles', methods=['POST'])
async def store_articles():
    """
    Store articles for a custom date range.
    Query Parameters:
        - from_date (str): Start date in 'YYYY-MM-DD' format.
        - to_date (str): End date in 'YYYY-MM-DD' format.
    """
    # Extract dates from JSON payload
    data = request.get_json()
    from_date = data.get('from_date')
    to_date = data.get('to_date')

    # Validate input
    if not from_date or not to_date:
        return jsonify({"error": "Missing 'from_date' or 'to_date' in request body"}), 400

    # Instantiate and invoke the StoreCustomRangeArticles class
    try:
        store_articles_handler = StoreCustomRangeArticles()
        result = await store_articles_handler.invoke(from_date=from_date, to_date=to_date)
        return jsonify(result)
    except Exception as e:
        print(f"Error in store_articles: {str(e)}")
        return jsonify({"error": str(e)}), 500





@app.route('/store_daily_articles', methods=['POST'])
async def store_daily_articles_route():
    """
    Async route to fetch and store daily articles.
    """
    try:
        # Instantiate the handler class
        store_articles_handler = StoreDailyArticles()
        
        # Invoke the class method
        result = await store_articles_handler.invoke()
        
        # Return JSON response
        return jsonify(result)
    except Exception as e:
        print(f"Error in /store_daily_articles: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
    


@app.route('/generate_questions', methods=['GET'])
def generate_questions_route():
    """
    Route to fetch articles and generate questions using imported functions.
    """
    try:
        # Call the imported function to fetch and generate questions
        results = fetch_questions_on_latest_articles_in_IndiaSpend()
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
