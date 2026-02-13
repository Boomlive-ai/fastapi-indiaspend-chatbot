import re, json, os
from langchain_core.messages import HumanMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Import necessary libraries
import nltk
print("NLTK PATH:", nltk.data.path)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_openai import OpenAIEmbeddings
import uuid
import requests
import os
import csv
from datetime import datetime
# import datetime
# print("date_time",datetime.__file__)


# Download the 'punkt' tokenizer resource
# nltk.download('punkt')

# Your other imports and code follow
from nltk.tokenize import word_tokenize


def get_qdrant_client():
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=False,
        timeout=30.0
    )


def create_collection_if_not_exists(collection_name="india-spend"):
    client = get_qdrant_client()

    existing = [c.name for c in client.get_collections().collections]

    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1536,  # text-embedding-3-small
                distance=Distance.COSINE
            )
        )
        print(f"Created collection: {collection_name}")

    return client

async def store_docs_in_qdrant(docs, collection_name="india-spend"):
    client = create_collection_if_not_exists(collection_name)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    texts = [doc.page_content for doc in docs]
    vectors = embeddings.embed_documents(texts)

    points = []

    for doc, vector in zip(docs, vectors):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": doc.page_content,
                    **doc.metadata
                }
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points
    )

    print(f"Stored {len(points)} chunks in Qdrant.")


def extract_sources_and_result(result: str):
    """
    Extracts URLs from the given result string and returns the result without sources.

    Args:
        result (str): The response content containing the sources.

    Returns:
        tuple: A tuple with two elements:
            1. result_without_sources (str): The content without the sources part.
            2. sources (list): A list of source URLs extracted from the result.
    """
    # Use regular expression to find all URLs in the result string
    sources = re.findall(r'https?://[^\s]+', result)
    
    # Remove all URLs from the result string
    result_without_sources = re.sub(r'https?://[^\s]+', '', result).strip()

    # Remove the "Sources" section and anything after it
    result_without_sources = re.sub(r'(Sources?:|References?:|See also:|Source:).*', '', result_without_sources).strip()

    # Remove any other potential sources-related headers (like "Related Articles", etc.)
    result_without_sources = re.sub(r'\s*(Sources?|References?|See also?):.*', '', result_without_sources).strip()

    # Remove leading or trailing whitespace after the replacement
    result_without_sources = result_without_sources.strip()

    # Remove duplicates from sources
    sources = list(set(sources))

    return result_without_sources, sources

def prioritize_sources(response_text: str, sources: list) -> list:
    """
    Reorder sources based on similarity to the response text.

    Args:
        response_text (str): The generated response content.
        sources (list): List of source URLs to prioritize.

    Returns:
        list: Reordered list of sources with the most relevant one at the top.
    """
    # If no response text or sources, return sources as is
    if not response_text or not sources:
        return sources

    # Combine response text and sources for comparison
    texts = [response_text] + sources  # Place response first
    vectorizer = TfidfVectorizer(stop_words="english")  # Use TF-IDF to vectorize
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Calculate cosine similarity between response_text and each source
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Sort sources by similarity scores in descending order
    sorted_indices = sorted(range(len(sources)), key=lambda i: similarities[i], reverse=True)
    sorted_sources = [sources[i] for i in sorted_indices][:5]

    return sorted_sources



def extract_last_human_message_and_sources(response: dict) -> tuple:
    """
    Fetch the last HumanMessage and extract sources from its content.

    Args:
        response (dict): The response containing messages with potential sources.

    Returns:
        tuple: A tuple containing the HumanMessage content and a list of source URLs.
    """
    last_human_message = None
    sources = []

    # Loop through messages in reverse to find the last HumanMessage
    for message in reversed(response.get("messages", [])):
        if isinstance(message, HumanMessage):
            last_human_message = message.content
            break

    # If a HumanMessage is found, extract sources
    if last_human_message:
        print("Last HumanMessage Content:\n", last_human_message)  # Debug: Print full content
        # Extract URLs using regex
        sources = re.findall(r'https?://[^\s]+', last_human_message)
        # Remove duplicates and clean the list
        sources = list(set(sources))
        sources = [source.strip() for source in sources]

    return  sources

def extract_clean_sources(response: dict) -> list:
    """
    Extract and clean source links from the last message in the response dictionary.

    Args:
        response (dict): The response containing messages with potential source links.

    Returns:
        list: A cleaned list of source URLs if found; otherwise, an empty list.
    """
    # Initialize an empty list to store source links
    sources = []

    # Get the list of messages
    messages = response.get("messages", [])

    # Check if there is at least one message
    if messages:
        # Get the last message
        last_message = messages[-1]

        # Access the 'content' attribute directly
        content = getattr(last_message, "content", "")
        print("Inspecting last message content:", content)  # Debugging line

        # Look for "Sources:" and extract URLs
        if "Sources:" in content:
            raw_sources = re.findall(r'https?://[^\s]+', content)
            sources.extend(raw_sources)

    # Remove duplicates and clean the list
    cleaned_sources = list(set(sources))  # Remove duplicates
    cleaned_sources = [source.strip() for source in cleaned_sources]  # Trim whitespace

    # Return the cleaned list (empty if no sources are found)
    return cleaned_sources



################################################VECTOR STORE DATABASE################################################################


# import datetime, json
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

async def store_daily_articles():
    """
    Fetch and store articles for the current day asynchronously.

    Returns:
        list: List of article URLs stored for the current day.
    """
    article_urls = []
    index_name = "india-spend"

    try:
        api_url = f'https://indiaspend.com/dev/h-api/news'
        headers = {
            "accept": "*/*",
            "s-id": os.getenv("INDIA_SPEND_S_ID")
        }
        print(f"Current API URL: {api_url}")

        response = requests.get(api_url, headers=headers)

        if response.status_code == 200:
            data = response.json()

            # Break if no articles are found
            if not data.get("news"):
                return
            for news_item in data.get("news", []):
                url_path = news_item.get("url")
                if url_path:
                    article_urls.append(url_path)
             # # Filter and process URLs
            filtered_urls = await filter_urls_custom_range(json.dumps(article_urls))
            # print("These are filtered urls",filtered_urls)
            docsperindex = await fetch_docs_custom_range(filtered_urls)
            print(f"Processed {len(filtered_urls)} articles and {len(docsperindex)} chunks to add to Pinecone.")

            # await store_docs_in_pinecone(docsperindex, index_name, filtered_urls)
            
            await store_docs_in_qdrant(docsperindex, index_name)
            await add_urls_to_database(json.dumps(filtered_urls))


            
        return filtered_urls
    except Exception as e:
        print(f"Error in store_daily_articles: {str(e)}")
        return []



# async def store_articles_custom_range(from_date: str = None, to_date: str = None):
#     """
#     Fetch and store articles based on a custom date range.

#     Args:
#         from_date (str): Start date in 'YYYY-MM-DD' format. Defaults to 6 months ago.
#         to_date (str): End date in 'YYYY-MM-DD' format. Defaults to today.

#     Returns:
#         list: List of all article URLs processed.
#     """
#     # Initialize variables
#     article_urls = []
#     start_index = 0
#     count = 20

#     # Calculate default date range if not provided
#     current_date = datetime.date.today()
#     if not to_date:
#         to_date = current_date.strftime('%Y-%m-%d')
#     if not from_date:
#         custom_months_ago = current_date - datetime.timedelta(days=180)  # Default to 6 months ago
#         from_date = custom_months_ago.strftime('%Y-%m-%d')

#     # Validate the date range
#     if not validate_date_range(from_date, to_date):
#         print("Invalid date range. Ensure 'from_date' <= 'to_date' and format is YYYY-MM-DD.")
#         return []

#     print(f"Fetching data from {from_date} to {to_date}....")
#     index_name = "india-spend"

#     while True:
#         perpageurl = []
#         print("Now start index is ", start_index)

#         # Construct API URL with the custom range
#         api_url = f'https://indiaspend.com/dev/h-api/news?startIndex={start_index}&count={count}&fromDate={from_date}&toDate={to_date}'
#         headers = {
#             "accept": "*/*",
#             "s-id": os.getenv("INDIA_SPEND_S_ID")
#         }
#         print(f"Current API URL: {api_url}")

#         response = requests.get(api_url, headers=headers)

#         if response.status_code == 200:
#             data = response.json()

#             # Break if no articles are found
#             if not data.get("news"):
#                 break

#             for news_item in data.get("news", []):
#                 url_path = news_item.get("url")
#                 if url_path:
#                     article_urls.append(url_path)
#                     perpageurl.append(url_path)

#             # print(perpageurl)
#             # # Filter and process URLs
#             filtered_urls = await filter_urls_custom_range(json.dumps(perpageurl))
#             # print("These are filtered urls",filtered_urls)
#             docsperindex = await fetch_docs_custom_range(filtered_urls)
#             print(f"Processed {len(filtered_urls)} articles and {len(docsperindex)} chunks to add to Pinecone.")

#             # await store_docs_in_pinecone(docsperindex, index_name, filtered_urls)
            
#             await store_docs_in_qdrant(docsperindex, index_name)
#             await add_urls_to_database(json.dumps(filtered_urls))

#             start_index += count
#         else:
#             print(f"Failed to fetch articles. Status code: {response.status_code}")
#             break

#     return article_urls

# async def store_articles_custom_range(from_date: str = None, to_date: str = None):
#     article_urls = []
#     start_index = 0
#     count = 20

#     current_date = datetime.date.today()

#     if not to_date:
#         to_date = current_date.strftime('%Y-%m-%d')

#     if not from_date:
#         custom_months_ago = current_date - datetime.timedelta(days=180)
#         from_date = custom_months_ago.strftime('%Y-%m-%d')

#     if not validate_date_range(from_date, to_date):
#         print("Invalid date range.")
#         return []

#     print(f"Fetching data from {from_date} to {to_date}...")
#     index_name = "india-spend"

#     while True:
#         perpageurl = []

#         # api_url = f'https://indiaspend.com/dev/h-api/news?startIndex={start_index}&count={count}&fromDate={from_date}&toDate={to_date}'
        
#         api_url = f'https://indiaspend.com/dev/h-api/news?fromDate={from_date}&toDate={to_date}'

#         headers = {
#             "accept": "*/*",
#             "s-id": os.getenv("INDIA_SPEND_S_ID")
#         }

#         response = requests.get(api_url, headers=headers)
#         date_news = response.json().get("date_news", [])
#         print("date",date_news)

#         if response.status_code != 200:
#             break

#         data = response.json()

#         if not data.get("news"):
#             break

#         # for news_item in data.get("news", []):
#         #     article_data = {
#         #         "url": news_item.get("url"),
#         #         "date_updated": news_item.get("date_updated"),
#         #         "date_news": news_item.get("date_news"),
#         #         "seo_title": news_item.get("seo_title"),
#         #         "description": news_item.get("description"),
#         #         "keywords": news_item.get("keywords"),
#         #     }
#         #     print("Article data:", article_data)  # Debugging line to inspect article data
#         #     url_path = news_item.get("url")
#         #     if url_path:
#         #         article_urls.append(url_path)
#         #         perpageurl.append(url_path)
#         for news_item in data.get("news", []):
#             article_data = {
#                 "url": news_item.get("url"),
#                 "date_updated": news_item.get("date_updated"),
#                 "date_news": news_item.get("date_news"),
#                 "seo_title": news_item.get("seo_title"),
#             }

#             if article_data["url"]:
#                 article_urls.append(article_data)
#                 perpageurl.append(article_data)

                
#         print("Raw URLs from API:", perpageurl)
#         print("Count before filtering:", len(perpageurl))

#         # filtered_urls = await filter_urls_custom_range(json.dumps(perpageurl))
        
#         filtered_urls = perpageurl
        
#         print("Filtered URLs:", filtered_urls)
#         print("Count after filtering:", len(filtered_urls))

#         docsperindex = await fetch_docs_custom_range(filtered_urls)

#         print(f"Processed {len(filtered_urls)} articles and {len(docsperindex)} chunks.")

#         # 🔥 Store in Qdrant instead of Pinecone
#         await store_docs_in_qdrant(docsperindex, index_name)

#         await add_urls_to_database(json.dumps(filtered_urls))

#         start_index += count

#     return article_urls



# async def store_articles_custom_range(from_date, to_date):
#     start_index = 0
#     count = 20
#     total_processed = 0
#     page_number = 1

#     log_file = "logs.csv"

#     # Create CSV file with header if not exists
#     if not os.path.exists(log_file):
#         with open(log_file, mode="w", newline="", encoding="utf-8") as file:
#             writer = csv.writer(file)
#             writer.writerow([
#                 "timestamp",
#                 "page_number",
#                 "start_index",
#                 "articles_received",
#                 "chunks_created",
#                 "total_processed",
#                 "status"
#             ])

#     while True:
#         api_url = (
#             f"https://indiaspend.com/dev/h-api/news"
#             f"?startIndex={start_index}"
#             f"&count={count}"
#             f"&fromDate={from_date}"
#             f"&toDate={to_date}"
#         )

#         print(f"\n📄 Fetching Page {page_number} | StartIndex: {start_index}")

#         try:
#             response = requests.get(
#                 api_url,
#                 headers={
#                     "accept": "*/*",
#                     "s-id": os.getenv("INDIA_SPEND_S_ID")
#                 }
#             )

#             if response.status_code != 200:
#                 print(f"❌ API failed with status {response.status_code}")
#                 status = f"API_ERROR_{response.status_code}"
#                 break

#             data = response.json()
#             news_list = data.get("news", [])

#             if not news_list:
#                 print("✅ No more articles found.")
#                 status = "NO_MORE_DATA"
#                 break

#             print(f"📰 Articles received: {len(news_list)}")

#             articles = []
#             for news_item in news_list:
#                 article_data = {
#                     "url": news_item.get("url"),
#                     "date_updated": news_item.get("date_updated"),
#                     "date_news": news_item.get("date_news"),
#                     "seo_title": news_item.get("seo_title"),
#                     "description": news_item.get("description"),
#                     "keywords": news_item.get("keywords"),
#                 }

#                 if article_data["url"]:
#                     articles.append(article_data)

#             docs = await fetch_docs_custom_range(articles)
#             chunks_created = len(docs)

#             print(f"✂ Chunks created: {chunks_created}")

#             await store_docs_in_qdrant(docs)

#             total_processed += len(articles)

#             status = "SUCCESS"

#             print(f"✅ Completed Page {page_number}")
#             print(f"📊 Total Articles Processed: {total_processed}")

#             # Write log to CSV
#             with open(log_file, mode="a", newline="", encoding="utf-8") as file:
#                 writer = csv.writer(file)
#                 writer.writerow([
#                     datetime.now().isoformat(),
#                     page_number,
#                     start_index,
#                     len(news_list),
#                     chunks_created,
#                     total_processed,
#                     status
#                 ])

#             # Move to next batch
#             start_index += count
#             page_number += 1

#         except Exception as e:
#             print(f"❌ Error occurred: {e}")
#             status = f"ERROR_{str(e)}"

#             with open(log_file, mode="a", newline="", encoding="utf-8") as file:
#                 writer = csv.writer(file)
#                 writer.writerow([
#                     datetime.now().isoformat(),
#                     page_number,
#                     start_index,
#                     0,
#                     0,
#                     total_processed,
#                     status
#                 ])
#             break

#     print("\n🎉 INGESTION COMPLETED")
#     print(f"📦 Total Articles Stored: {total_processed}")

async def store_articles_custom_range(from_date, to_date):
    start_index = 0
    count = 20
    total_processed = 0
    page_number = 1
    # log_file = "logs.csv"

    # # Create CSV file with header if it doesn't exist
    # if not os.path.exists(log_file):
    #     with open(log_file, mode="w", newline="", encoding="utf-8") as file:
    #         writer = csv.writer(file)
    #         writer.writerow([
    #             "timestamp",
    #             "page_number",
    #             "start_index",
    #             "articles_received",
    #             "chunks_created",
    #             "total_processed",
    #             "status"
    #         ])

    while True:
        api_url = (
            f"https://indiaspend.com/dev/h-api/news"
            f"?startIndex={start_index}"
            f"&count={count}"
            f"&fromDate={from_date}"
            f"&toDate={to_date}"
        )

        print(f"\n📄 Fetching Page {page_number} | StartIndex: {start_index}")

        try:
            response = requests.get(
                api_url,
                headers={
                    "accept": "*/*",
                    "s-id": os.getenv("INDIA_SPEND_S_ID")
                }
            )

            if response.status_code != 200:
                status = f"API_ERROR_{response.status_code}"
                print(f"❌ API failed: {status}")
                break

            data = response.json()
            news_list = data.get("news", [])

            if not news_list:
                print("✅ No more articles found.")
                status = "NO_MORE_DATA"
                break

            print(f"📰 Articles received: {len(news_list)}")

            articles = []
            for news_item in news_list:
                article_data = {
                    "url": news_item.get("url"),
                    "date_updated": news_item.get("date_updated"),
                    "date_news": news_item.get("date_news"),
                    "seo_title": news_item.get("seo_title"),
                    "description": news_item.get("description"),
                    "keywords": news_item.get("keywords"),
                }

                if article_data["url"]:
                    articles.append(article_data)

            # Fetch + preprocess
            docs = await fetch_docs_custom_range(articles)
            chunks_created = len(docs)

            print(f"✂ Chunks created: {chunks_created}")

            # Store in Qdrant
            await store_docs_in_qdrant(docs)

            total_processed += len(articles)
            status = "SUCCESS"

            print(f"✅ Completed Page {page_number}")
            print(f"📊 Total Articles Processed: {total_processed}")

            # Write success log
            # with open(log_file, mode="a", newline="", encoding="utf-8") as file:
            #     writer = csv.writer(file)
            #     writer.writerow([
            #         datetime.now().isoformat(),
            #         page_number,
            #         start_index,
            #         len(news_list),
            #         chunks_created,
            #         total_processed,
            #         status
            #     ])

            # Move to next batch
            start_index += count
            page_number += 1

        except Exception as e:
            status = f"ERROR: {str(e)}"
            print(f"❌ Error occurred: {status}")

            # with open(log_file, mode="a", newline="", encoding="utf-8") as file:
            #     writer = csv.writer(file)
            #     writer.writerow([
            #         datetime.now().isoformat(),
            #         page_number,
            #         start_index,
            #         0,
            #         0,
            #         total_processed,
            #         status
            #     ])

            break

    print("\n🎉 INGESTION COMPLETED")
    print(f"📦 Total Articles Stored: {total_processed}")





def validate_date_range(from_date: str, to_date: str) -> bool:
    """
    Validate the custom date range.

    Args:
        from_date (str): Start date.
        to_date (str): End date.

    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        from_dt = datetime.datetime.strptime(from_date, '%Y-%m-%d')
        to_dt = datetime.datetime.strptime(to_date, '%Y-%m-%d')
        return from_dt <= to_dt
    except ValueError:
        return False


async def filter_urls_custom_range(urls):
    api_url = f"https://exceltohtml.indiaspend.com/chatbotDB/not_in_table.php?urls={urls}"
    headers = {
        "accept": "*/*",
        "Authorization": os.getenv("AUTHORIZATION_TOKEN"),
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(api_url, headers=headers, verify=False)
        if response.status_code == 200:
            response_data = response.json()
            return response_data.get("urls", [])
    except requests.RequestException as e:
        print(f"Error filtering URLs: {e}")
    return []




# async def fetch_docs_custom_range(urls):
#     data = []
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

#     for url in urls:
#         try:
#             response = requests.get(url, timeout=10)
#             response.raise_for_status()

#             # Parse only HTML content
#             if 'text/html' not in response.headers.get('Content-Type', ''):
#                 print(f"Skipped non-HTML content at {url}")
#                 continue

#             # Extract text using BeautifulSoup
#             soup = BeautifulSoup(response.content, 'html.parser')
#             text = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3'])])

#             document = Document(page_content=text, metadata={"source": url})
#             data.append(document)
#         except requests.exceptions.RequestException as e:
#             print(f"Failed to fetch {url}: {e}")
#             continue

#     docs = text_splitter.split_documents(data)
#     preprocessed_docs = await preprocess_documents(docs)

#     return preprocessed_docs

async def fetch_docs_custom_range(articles):
    data = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    for article in articles:
        try:
            url = article.get("url")
            date_updated = article.get("date_updated")
            date_news = article.get("date_news")
            seo_title = article.get("seo_title")
            description = article.get("description")
            keywords = article.get("keywords")

            if not url:
                continue

            response = requests.get(url, timeout=10)
            response.raise_for_status()

            # Ensure we only parse HTML
            if 'text/html' not in response.headers.get('Content-Type', ''):
                print(f"Skipped non-HTML content at {url}")
                continue

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract text content
            text = ' '.join(
                [p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3'])]
            )

            document = Document(
                page_content=text,
                metadata={
                    "source": url,
                    "seo_title": seo_title,
                    "date_updated": date_updated,
                    "date_news": date_news,
                    "description": description,
                    "keywords": keywords
                }
            )

            data.append(document)

        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch {url}: {e}")
            continue

    # Split into chunks
    docs = text_splitter.split_documents(data)

    # Optional preprocessing (if you already have this function)
    preprocessed_docs = await preprocess_documents(docs)

    return preprocessed_docs



async def store_docs_in_pinecone(docs, index_name, urls):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print(f"Storing {len(docs)} document chunks to Pinecone index '{index_name}'...")
    pine_vs = Pinecone.from_documents(documents = docs, embedding = embeddings, index_name=index_name)
    print(f"Added {len(docs)} Articles chunks in the pinecone")
    await add_urls_to_database(json.dumps(urls))
    print(f"Successfully stored documents. Associated URLs: {urls}")
    return pine_vs



async def add_urls_to_database(urls):
    """
    Adds new URLs to the database by sending them to an external API endpoint.

    Args:
        urls (list): List of new URLs to be added to the database.

    Returns:
        str: A message indicating the result of the request.
    """
    api_url = f"https://exceltohtml.indiaspend.com/chatbotDB/add_in_table.php?urls={urls}"
    headers = {
        "accept": "*/*",
        "Authorization": os.getenv("AUTHORIZATION_TOKEN"),
        "Content-Type": "application/json"
    }
    
    try:
        # Send the POST request with the URLs in the payload
        response = requests.get(api_url, headers=headers, verify=False)

        # Check if the request was successful
        if response.status_code == 200:
            response_data = response.json()
            # You can log or process the response data as required
            # noofurls = len(urls)
            # print(urls, noofurls)
            print(f"Successfully added {len(urls)}URLs to the database." )
            return f"Successfully added URLs to the database."
        else:
            if(len(urls) == 0):
                return f"There are no urls to add"
            return f"There are no urls to add"
    except requests.RequestException as e:
        return f"An error occurred while adding URLs: {e}"
    

#######################################################################Developers FUcntionss####################################################################################
async def process_and_upload_single_url(url, index_name="india-spend"):
    """
    Process a single URL: fetch content, convert to document, preprocess, and upload to Pinecone.
    
    Args:
        url (str): The URL of the article to process and upload
        index_name (str): Name of the Pinecone index to use (default: "india-spend")
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Processing URL: {url}")
        
        # 1. Fetch content from the URL
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Check if it's HTML content
        if 'text/html' not in response.headers.get('Content-Type', ''):
            print(f"Skipped non-HTML content at {url}")
            return False
            
        # 2. Extract text using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        text = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3'])])
        
        # 3. Create document
        document = Document(page_content=text, metadata={"source": url})
        
        # 4. Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        doc_chunks = text_splitter.split_documents([document])
        
        # 5. Preprocess document chunks
        preprocessed_docs = await preprocess_documents(doc_chunks)
        
        # 6. Upload to Pinecone
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        print(f"Storing {len(preprocessed_docs)} document chunks to Pinecone index '{index_name}'...")
        
        pine_vs = Pinecone.from_documents(
            documents=preprocessed_docs, 
            embedding=embeddings, 
            index_name=index_name
        )
        
        # 7. Add URL to database
        await add_urls_to_database(json.dumps([url]))
        
        print(f"Successfully processed and uploaded content from {url}")
        return True
        
    except Exception as e:
        print(f"Error processing URL {url}: {str(e)}")
        return False



def preprocess_query(query_text):
    """
    Preprocess a query string using the same steps as document preprocessing.
    
    Args:
        query_text (str): The raw query text
        
    Returns:
        str: Preprocessed query text
    """
    # Download required NLTK packages if not already downloaded
    try:
        
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        # Make sure to download these resources first
        
        # nltk.download('punkt')
        # nltk.download('punkt_tab')  # though this might not exist as a standard resource
        # nltk.download('stopwords')
        # nltk.download('wordnet')
    except Exception as e:
        print(f"Error downloading NLTK resources: {str(e)}")
    
    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # 1. Basic cleaning
    # Remove URLs
    query_text = re.sub(r'https?://\S+|www\.\S+', '', query_text)
    # Remove HTML tags
    query_text = re.sub(r'<.*?>', '', query_text)
    # Remove special characters and numbers
    query_text = re.sub(r'[^a-zA-Z\s]', '', query_text)
    # Normalize whitespace
    query_text = ' '.join(query_text.split())
    
    # 2. Tokenize the text
    tokens = word_tokenize(query_text.lower())
    
    # 3. Remove stopwords and lemmatize
    filtered_tokens = []
    for token in tokens:
        if token not in stop_words:
            # Lemmatize the token
            lemmatized = lemmatizer.lemmatize(token)
            filtered_tokens.append(lemmatized)
    
    # 4. Reconstruct the text
    processed_query = ' '.join(filtered_tokens)
    
    return processed_query

async def query_pinecone(query_text, index_name="india-spend", top_k=5):
    """
    Query the Pinecone index with preprocessing.
    
    Args:
        query_text (str): The raw query text
        index_name (str): Name of the Pinecone index to use
        top_k (int): Number of results to return
        
    Returns:
        list: List of document results from Pinecone
    """


    # Preprocess the query
    processed_query = preprocess_query(query_text)
    
    # Generate embeddings using the same model as for documents
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Connect to Pinecone
    pine = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
    
    # Query the index
    results = pine.similarity_search(processed_query, k=top_k)
    
    return results


#######################################################################Developers FUcntionss####################################################################################

####################################################################### Preprocessing the DOCUMENTS ###################################################



import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

async def preprocess_documents(docs):
    """
    Comprehensive preprocessing of documents before storing them in Pinecone.
    
    Steps include:
    - Text cleaning (whitespace normalization, special character removal)
    - Stopword removal
    - Lemmatization (converting words to their base form)
    - Additional metadata
    
    Args:
        docs (list): List of Document objects to preprocess
        
    Returns:
        list: List of preprocessed Document objects
    """
    # Download required NLTK packages if not already downloaded
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        raise RuntimeError("NLTK wordnet not installed. Please install before running.")
        # nltk.download('punkt')
        # nltk.download('stopwords')
        # nltk.download('wordnet')
    
    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    preprocessed_docs = []
    
    for doc in docs:
        # Extract content from the document
        content = doc.page_content
        metadata = doc.metadata.copy()  # Create a copy to avoid modifying the original
        
        # 1. Basic cleaning
        # Remove URLs
        content = re.sub(r'https?://\S+|www\.\S+', '', content)
        # Remove HTML tags
        content = re.sub(r'<.*?>', '', content)
        # Remove special characters and numbers (keep spaces and word characters)
        content = re.sub(r'[^a-zA-Z\s]', '', content)
        # Normalize whitespace
        content = ' '.join(content.split())
        
        # 2. Tokenize the text
        tokens = word_tokenize(content.lower())
        
        # 3. Remove stopwords and lemmatize
        filtered_tokens = []
        for token in tokens:
            if token not in stop_words:
                # Lemmatize the token (e.g., "richest" -> "rich")
                lemmatized = lemmatizer.lemmatize(token)
                filtered_tokens.append(lemmatized)
        
        # 4. Reconstruct the text
        processed_content = ' '.join(filtered_tokens)
        
        # 5. Add preprocessing metadata
        metadata["preprocessed"] = True
        metadata["processed_date"] = datetime.now().isoformat()
        metadata["original_length"] = len(content)
        metadata["processed_length"] = len(processed_content)
        
        # Create a new document with the preprocessed content
        preprocessed_doc = Document(page_content=processed_content, metadata=metadata)
        preprocessed_docs.append(preprocessed_doc)
    
    print(f"Preprocessed {len(preprocessed_docs)} document chunks (removed stopwords, applied lemmatization)")
    return preprocessed_docs