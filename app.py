import os, json, re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from bot import Chatbot
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessageChunk
from utils import extract_sources_and_result, prioritize_sources
from tools import fetch_questions_on_latest_articles_in_IndiaSpend, extract_links, generate_questions_for_articles
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
@app.get("/store_daily_articles")
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




###########################################################################DEVELOPERS ROUTES#############################################################################
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from utils import process_and_upload_single_url, query_pinecone
class UrlItem(BaseModel):
    url: str

class QueryItem(BaseModel):
    query: str
    top_k: Optional[int] = 5

class ProcessResponse(BaseModel):
    success: bool
    message: str

# Route for processing a single URL
@app.post("/process-url", response_model=ProcessResponse)
async def process_url(url_item: UrlItem, background_tasks: BackgroundTasks):
    """
    Process and upload a single URL to the Pinecone index
    """
    try:
        # Add the task to background tasks to not block the response
        background_tasks.add_task(process_and_upload_single_url, url_item.url)
        return {"success": True, "message": f"URL processing started for: {url_item.url}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing URL: {str(e)}")

# Route for querying the Pinecone index
@app.post("/query_pinecone")
async def query_index(query_item: QueryItem):
    """
    Query the Pinecone index with the given query text
    """
    try:
        results = await query_pinecone(query_item.query, top_k=query_item.top_k)
        # Format results for response
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "source": doc.metadata.get("source", "Unknown"),
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        return {"success": True, "results": formatted_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying index: {str(e)}")
    
class LinkExtractorResponse(BaseModel):
    links: List[str]
    count: int
    url: str

# Add this route to your FastAPI app
@app.get("/extract_links", response_model=LinkExtractorResponse)
async def get_extract_links(url_or_path: str):
    """
    Extract article links from an IndiaSend webpage.
    
    Parameters:
    - url_or_path: Either a full URL (https://www.indiaspend.com/earthcheckindia) 
                  or just the path segment (earthcheckindia)
    
    Returns:
    - List of article URLs extracted from the page
    - Count of extracted links
    - Full URL that was processed
    """
    try:
        links, full_url = extract_links(url_or_path)
        return {
            "links": links,
            "count": len(links),
            "url": full_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting links: {str(e)}")

   ###########################################################################DEVELOPERS ROUTES#############################################################################


# FastAPI Route for extracting links and generating questions
@app.get("/generate_tags_questions", response_model=dict)
async def extract_and_generate(url_or_path: str):
    """
    Extracts links from a webpage and generates relevant questions.
    
    Parameters:
    - url_or_path: Either a full URL (https://www.indiaspend.com/earthcheckindia) 
                   or just the path segment (earthcheckindia)
    
    Returns:
    - List of extracted article links
    - List of generated questions
    - Count of extracted links
    - Full processed URL
    """
    try:
        result = generate_questions_for_articles(url_or_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    

    
import uvicorn

if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


