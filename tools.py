import re
import requests
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

# Initialize the LLM
llm = ChatOpenAI(temperature=0, model_name='gpt-4o')

def generate_questions_batch(articles):
    """
    Generates specific, concise questions for a batch of articles using an LLM,
    focusing on keywords and actionable information. The questions are returned 
    randomly arranged and cleaned up.

    Parameters:
        articles (list): List of article dictionaries.

    Returns:
        list: A list of questions randomly arranged and cleaned up.
    """
    # Construct a single prompt for all articles in the batch
    input_prompts = []
    
    for i, article in enumerate(articles):
        title = article.get("heading", "Untitled Article")
        description = article.get("description", "No description available.")
        story = article.get("story", "No story content available.")
        url = article.get("url")
        # Extract keywords from the article description and story for more specific question generation
        keywords = list(set(re.findall(r'\b\w+\b', description + " " + story.lower())))[:10]  # Convert set to list and slice first 10 words
        
        input_prompts.append(f"""
        Title: {title}
        Description: {description}
        Story Excerpt: {story[:500]}... (truncated for brevity)
        Keywords: {', '.join(keywords)}
        Url: {url}
        Generate two concise,interesting, specific and relevant questions (under 60 characters) based on the article content which user want to know about latest articles.
        Ensure the questions meet the following criteria:
        1. Focus on actionable or data-driven information from the article.
        2. Do not include the article title, article labels, or headings in the questions.
        3. Do not use bullet points or article numbers.
        4. Return only the questions (no introductory text or labels).
        5. Keep the questions under 60 characters each.
        6. Should have a emoji related to question as prefix
        7. Return the questions in a shuffled order.
        """)



    # Combine all prompts into one input
    batch_prompt = "\n".join(input_prompts)

    try:
        # Create a HumanMessage object for the LLM
        message = HumanMessage(content=batch_prompt)

        # Invoke the LLM with the message
        response = llm.invoke([message])

        # Split the response into individual lines (questions)
        questions = response.content.strip().split("\n")

        # Remove any empty strings from the list of questions
        cleaned_questions = [q.strip() for q in questions if q.strip()]

        return cleaned_questions

    except Exception as e:
        print(f"Error generating questions: {e}")
        return []



def fetch_questions_on_latest_articles_in_IndiaSpend():
    """
    Fetches the latest articles from the IndiaSpend API and generates up to 20
    concise questions in batches.

    Returns:
        dict: A dictionary containing all questions from the articles in a single list.
    """
    api_url = 'https://indiaspend.com/dev/h-api/news'
    headers = {
        "accept": "*/*",
        "s-id": "yP4PEm9PqCPxxiMuHDyYz0PIjAHxHbYpTQi9E4AtNk0R4bp9Lsf0hyN4AEKKiM9I"
    }
    print(f"Fetching articles from API: {api_url}")

    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch articles: {e}")
        return {"error": f"Failed to fetch articles: {e}"}

    articles = data.get("news", [])
    if not articles:
        print("No articles found in the response.")
        return {"questions": []}

    # Limit articles to 10 (as each article generates 2 questions)
    articles = articles[:10]

    # Generate questions in a single batch
    questions = generate_questions_batch(articles)

    # Ensure only 20 questions are returned
    return {"questions": questions[:20]}




import requests
from bs4 import BeautifulSoup
import re
def extract_links(url_or_path):
    """
    Extract href links from IndiaSend website given a URL or path.
    
    Returns:
    tuple: (list of article URLs, full URL processed)
    """
    base_url = "https://www.indiaspend.com"
    
    if not url_or_path.startswith("http"):
        if url_or_path.startswith("/"):
            url_or_path = url_or_path[1:]
        full_url = f"{base_url}/{url_or_path}"
    else:
        full_url = url_or_path
    
    try:
        response = requests.get(full_url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        article_links = []

        for article in soup.find_all('article'):
            title_element = article.find('h3', class_='alith_post_title')
            if title_element and title_element.find('a'):
                href = title_element.find('a')['href']
                if href.startswith('http'):
                    if 'indiaspend.com' not in href:
                        continue
                    article_links.append(href)
                else:
                    if href.startswith('/'):
                        article_links.append(f"{base_url}{href}")
                    else:
                        article_links.append(f"{base_url}/{href}")
        
        return article_links, full_url  # ✅ Now returning a tuple

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return [], full_url  # ✅ Return empty list and full_url in case of error
    except Exception as e:
        print(f"Error parsing the HTML: {e}")
        return [], full_url


# Function to generate questions from extracted links
def generate_questions_for_articles(url_or_path):
    links, full_url = extract_links(url_or_path)

    if not links:
        return {
            "links": [],
            "questions": [],
            "count": 0,
            "url": full_url
        }
    
    # Simulating fetching article content (you need real API calls or scraping here)
    articles = []
    for link in links:
        # Dummy data, replace with actual content fetching logic
        articles.append({
            "heading": f"Article from {link}",
            "description": "This is a brief description of the article.",
            "story": "This is a placeholder for the article's main content.",
            "url": link
        })
    
    # Generate questions based on articles
    questions = generate_questions_batch(articles)

    return {
        "links": links,
        "questions": questions,
        "count": len(links),
        "url": full_url
    }

