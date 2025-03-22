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
