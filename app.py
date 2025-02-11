import os, json, re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from bot import Chatbot
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessageChunk
from utils import extract_sources_and_result, prioritize_sources
from tools import fetch_questions_on_latest_articles_in_IndiaSpend
from vectorstore import StoreCustomRangeArticles, StoreDailyArticles
from fastapi import FastAPI, HTTPException


# Initialize FastAPI
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize environment variables
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")
os.environ['OPENAI_API_KEY']= os.getenv("OPENAI_API_KEY")
# Initialize chatbot
mybot = Chatbot()
workflow = mybot()

# Request models
class ArticleRangeRequest(BaseModel):
    from_date: str
    to_date: str

 
    
@app.get("/stream_query")
async def stream_query_bot(question: str, thread_id: str):
    if not question or not thread_id:
        raise HTTPException(status_code=400, detail="Missing required parameters.")
    
    input_data = {"messages": [HumanMessage(content=question)]}

    async def stream_chunks():
        sources = []
        try:
            async for event in workflow.astream_events(input_data, config={"configurable": {"thread_id": thread_id}}, version="v2"):
                # print(event["event"])

                if event["event"]=="on_chat_model_end":
                    # print(event["data"])
                    pass

                if event["event"]=="on_retriever_end":
                    print("*************************************************************************")
                    # print(event["data"])  # Debug print to check the data structure

                    # Extract sources from the "output" field
                    output = event["data"].get("output", [])
                    if isinstance(output, list):  # Ensure the output is a list
                        sources.extend(
                            [doc.metadata.get("source") for doc in output if doc.metadata.get("source")]
                        )  # Print the extracted sources
                       

                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    # print(chunk.content, end="|", flush=True)
                    if isinstance(chunk, AIMessageChunk):
                        # print(chunk)
                        match = re.search(r"content='([^']+)'", str(chunk))
                        if match:
                            content = match.group(1)
                            # content = content.replace('\n', '<br>')
                            # content = content.replace('.\n\n', '.<br><br>')
                            yield f"data: {content}\n\n"  # Format for SSE

                    else:
                        yield "data: Invalid chunk type\n\n"
        except Exception as e:
            yield f"data: Error in query_bot: {str(e)}\n\n"
        finally:
        # Send the end marker when the stream finishes
            if sources:
                sources = prioritize_sources(question, sources)
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                print("QUESTION", question)
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                print("SOURCES", sources)
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

                yield f"data: {json.dumps({'sources': sources})}\n\n"
            yield "data: [end]\n\n"

    return StreamingResponse(
        stream_chunks(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/query")
async def query_bot(question: str, thread_id: str):
    if not question or not thread_id:
        raise HTTPException(status_code=400, detail="Missing required parameters.")
    
    input_data = {"messages": [HumanMessage(content=question)]}
    try:
        response = workflow.invoke(input_data, config={"configurable": {"thread_id": thread_id}})
        result = response['messages'][-1].content
        result, raw_sources = extract_sources_and_result(result)
        sources = prioritize_sources(result, raw_sources)
        if not result:
            result = "No response generated. Please try again."
        return {"response": result, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in query_bot: {str(e)}")

# Store articles for a custom date range
@app.post("/store_articles")
async def store_articles(data: ArticleRangeRequest):
    try:
        store_articles_handler = StoreCustomRangeArticles()
        result = await store_articles_handler.invoke(from_date=data.from_date, to_date=data.to_date)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in store_articles: {str(e)}")

# Store daily articles
@app.post("/store_daily_articles")
async def store_daily_articles():
    try:
        store_articles_handler = StoreDailyArticles()
        result = await store_articles_handler.invoke()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in store_daily_articles: {str(e)}")

# Generate questions from latest articles
@app.get("/generate_questions")
async def generate_questions():
    try:
        results = fetch_questions_on_latest_articles_in_IndiaSpend()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in generate_questions: {str(e)}")

# Documentation endpoint
@app.get("/")
async def documentation():
    return {
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
        ]
    }
   
import uvicorn

if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")