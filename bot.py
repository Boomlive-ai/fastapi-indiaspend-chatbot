from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv
from utils import preprocess_query, get_qdrant_client  # Import your Qdrant client
load_dotenv()
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
os.environ['OPENAI_API_KEY']= os.getenv("OPENAI_API_KEY")

class RAGQuery(BaseModel):
    query: str = Field(..., description="The query to retrieve relevant content for")

class RAGTool:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        # self.index_name = "india-spend"
        # self.vectorstore = PineconeVectorStore(
        #     index_name=self.index_name,
        #     embedding=self.embeddings
        # )
        # Initialize Qdrant client from utils
        self.qdrant_client = get_qdrant_client()
        
        # Initialize Qdrant vector store with LangChain
        self.vectorstore = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name="india-spend-1",
            embedding=self.embeddings
        )
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=False)
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            # search_kwargs={"k": 20, "lambda_mult": 0.5, "fuzzy": True}
            search_kwargs={"k": 8, "lambda_mult": 0.5}
        )
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="map_reduce",
            retriever=self.retriever,
            return_source_documents=True  # Enable source document return
        )
        
    
    def title_similarity(self, query: str, title: str) -> float:
        """
        Compute embedding cosine similarity between query and title
        """
        embeddings = self.embeddings.embed_documents([query, title])

        similarity = cosine_similarity(
            [embeddings[0]],
            [embeddings[1]]
        )[0][0]

        return float(similarity)
    
    def filter_by_title_match(self, docs, query, threshold=0.90):
        """
        Return documents where title similarity >= threshold
        """
        matched_docs = []

        for doc in docs:
            title = doc.metadata.get("title", "")

            if not title:
                continue

            sim = self.title_similarity(query, title)

            print(f"Title: {title}")
            print(f"Similarity: {sim}")

            if sim >= threshold:
                matched_docs.append(doc)

        return matched_docs
    
    def rerank_documents(self, docs, query):
        query_terms = query.lower().split()
        now = datetime.now()

        ranked = []

        for doc in docs:
            relevance_score = 0
            content = doc.page_content.lower()
            title = doc.metadata.get("title", "").lower()

            # -----------------------------
            # Keyword relevance scoring
            # -----------------------------
            for term in query_terms:
                if term in title:
                    relevance_score += 3
                if term in content:
                    relevance_score += 1

            # -----------------------------
            # Recency scoring
            # -----------------------------
            recency_score = 0
            raw_date = doc.metadata.get("date_news")

            if raw_date:
                try:
                    doc_date = datetime.strptime(raw_date, "%Y-%m-%d %H:%M:%S.%f")
                except:
                    try:
                        doc_date = datetime.strptime(raw_date.split(" ")[0], "%Y-%m-%d")
                    except:
                        doc_date = None

                if doc_date:
                    days_old = (now - doc_date).days

                    if days_old <= 7:
                        recency_score = 4
                    elif days_old <= 30:
                        recency_score = 3
                    elif days_old <= 180:
                        recency_score = 2
                    elif days_old <= 365:
                        recency_score = 1

            final_score = relevance_score + recency_score

            ranked.append((final_score, doc))

        ranked.sort(key=lambda x: x[0], reverse=True)

        return [doc for score, doc in ranked if score > 0]


    # def retrieve(self, query: RAGQuery) -> dict:
    #     # print(f"Retrieving for query: {query.query}")
    #     similar_docs = self.retriever.get_relevant_documents(query.query)
    #     print("query", query)
    #     source_links = [doc.metadata.get('source', 'No source') for doc in similar_docs]
    #     print("metadata", source_links)
        
    #     # result = self.rag_chain.invoke(query.query + "Note do not mention anything about the provided context")
    #     # if isinstance(result, dict):
    #     #     response_result = result.get('result', str(result))
    #     #     source_documents = result.get('source_documents', [])
    #     #     source_links.extend([doc.metadata.get('source', 'No source') for doc in source_documents])
    #     # else:
    #     #     response_result = str(result)
            
    #     # Remove duplicates while preserving order
    #     source_links = list(dict.fromkeys(source_links))
        
    #     return {
    #         # "result": response_result,
    #         "sources": source_links
    #     }
    
    # def retrieve(self, query: RAGQuery) -> dict:
    #     # Step 1: Initial semantic retrieval
    #     similar_docs = self.retriever.get_relevant_documents(query.query)
    #     print("Initial Retrieved Docs:", [doc.metadata.get("source") for doc in similar_docs])
    #     for doc in similar_docs:
    #         print("PAGE CONTENT:", doc.page_content[:200])
    #         print("METADATA:", doc.metadata)
    #         print("-" * 50)


    #     # Step 2: Re-rank documents for accuracy
    #     reranked_docs = self.rerank_documents(similar_docs, query.query)

    #     # Step 3: Extract sources from reranked docs
    #     source_links = [
    #         doc.metadata.get("source", "No source")
    #         for doc in reranked_docs
    #     ][:5]

    #     # Step 4: Remove duplicates, preserve order
    #     source_links = list(dict.fromkeys(source_links))
    #     print("Reranked Sources:", source_links)    

    #     return {
    #         "sources": source_links,
    #         "docs": reranked_docs   # 👈 ADD THIS
    #     }
    
    def check_relevance(self, query: str, docs, threshold=0.45) -> bool:
        """
        Check if any retrieved document is actually relevant to the query.
        Uses embedding cosine similarity between query and top doc contents.
        Returns True if at least one doc meets the relevance threshold.
        """
        if not docs:
            return False

        query_embedding = self.embeddings.embed_query(query)

        for doc in docs[:3]:  # check top 3 docs
            content = doc.page_content[:1000]  # first 1000 chars
            title = doc.metadata.get("title", "")
            text = f"{title} {content}"

            doc_embedding = self.embeddings.embed_query(text)

            similarity = cosine_similarity(
                [query_embedding], [doc_embedding]
            )[0][0]

            print(f"📊 Relevance check — Title: {title[:60]}... | Similarity: {similarity:.4f}")

            if similarity >= threshold:
                return True

        return False

    def retrieve(self, query: RAGQuery) -> dict:

        # Step 1: semantic retrieval
        similar_docs = self.retriever.get_relevant_documents(query.query)

        print("Initial Retrieved Docs:", [doc.metadata.get("source") for doc in similar_docs])

        # Step 2: Title similarity check (>=90%)
        title_matched_docs = self.filter_by_title_match(similar_docs, query.query, threshold=0.90)

        if title_matched_docs:
            print("✅ Title matched docs found")
            reranked_docs = title_matched_docs

        else:
            print("⚠️ No strong title match, using normal reranking")
            reranked_docs = self.rerank_documents(similar_docs, query.query)

        # Step 3: Relevance gate — check if docs are actually related to the query
        is_relevant = self.check_relevance(query.query, reranked_docs, threshold=0.45)

        if not is_relevant:
            print("❌ No relevant documents found for this query")
            return {
                "sources": [],
                "docs": [],
                "is_relevant": False
            }

        print("✅ Relevant documents found")

        # Step 4: Extract sources
        source_links = [
            doc.metadata.get("source", "No source")
            for doc in reranked_docs
        ][:5]

        source_links = list(dict.fromkeys(source_links))

        return {
            "sources": source_links,
            "docs": reranked_docs,
            "is_relevant": True
        }

    
    def retrieve_from_sources(self, query: str, source_links: list[str]) -> list[str]:
        """
        Re-rank using semantic retrieval, but keep only provided source_links
        """
        # Run normal retrieval + reranking
        similar_docs = self.retriever.get_relevant_documents(query)
        reranked_docs = self.rerank_documents(similar_docs, query)

        reranked_sources = [
            doc.metadata.get("source")
            for doc in reranked_docs
            if doc.metadata.get("source") in source_links
        ]

        # Deduplicate + limit
        return list(dict.fromkeys(reranked_sources))[:5]

    def retrieve_chunk_for_bold_phrase(self, phrase: str, k=3):
        """
        Semantic retrieval for a bold phrase to find
        the most relevant document chunk.
        """
        results = self.vectorstore.similarity_search_with_score(
            phrase,
            k=k
        )

        # lower score = better similarity in Pinecone
        best_doc, best_score = results[0]

        return best_doc, best_score
    
    def extract_query_context(self, query: str) -> str:
        """
        Extracts a short semantic intent/context of the user query.
        """
        prompt = f"""
        Extract the core topic and intent of the query below in 1–2 sentences.
        Do NOT add extra explanation.

        Query:
        {query}
        """

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()


    def extract_context_window(self, doc, phrase, window=40):
        """
        Returns a small context window around the phrase.
        """
        content = doc.page_content
        content_lower = content.lower()
        phrase_lower = phrase.lower()

        idx = content_lower.find(phrase_lower)
        if idx == -1:
            return content[:300]  # fallback

        start = max(0, idx - window * 5)
        end = min(len(content), idx + len(phrase) + window * 5)

        return content[start:end].strip()
    
    

    def context_similarity(self, query_context: str, doc_context: str) -> float:
        """
        Returns cosine similarity between query context and document context.
        """
        embeddings = self.embeddings.embed_documents(
            [query_context, doc_context]
        )

        sim = cosine_similarity(
            [embeddings[0]],
            [embeddings[1]]
        )[0][0]

        return round(float(sim), 2)


    # def map_bold_phrases_to_sources(self, bold_phrases, confidence_threshold=0.50):
    #     print("Mapping bold phrases to sources with threshold:", confidence_threshold)
    #     mappings = {}

    #     for phrase in bold_phrases:
    #         doc, score = self.retrieve_chunk_for_bold_phrase(phrase)
    #         print("PHRASE:", phrase)
    #         print("RAW SCORE:", score)

    #         # # Pinecone similarity score: lower is better
    #         # similarity = 1 - score  

    #         # if similarity < confidence_threshold:
    #         #     continue
            
    #         # # ✅ Keep only strong semantic matches
    #         # if score <= 0.50:
    #         #     continue
            
    #         if score < confidence_threshold:  # default threshold = 0.60
    #             print(f"⚠️ Score {score} below threshold {confidence_threshold}, skipping")
    #             continue

    #         context = self.extract_context_window(doc, phrase)

    #         mappings[phrase] = {
    #             "source": doc.metadata.get("source"),
    #             "confidence": round(score, 2),
    #             "context": context
    #         }
    #         print(f"Mapped '{phrase}' to {doc.metadata.get('source')} with confidence {round(score, 2)}")
            


    #     return mappings
    
    # def map_bold_phrases_to_sources(
    #     self,
    #     bold_phrases,
    #     query: str,
    #     confidence_threshold=0.5,
    #     context_threshold=0.5
    # ):
    def _title_match_percentage(self, phrase: str, title: str) -> float:
        """
        Calculate what percentage of the phrase tokens appear in the title.
        Returns a value between 0 and 100.
        """
        phrase_tokens = phrase.lower().strip().split()
        title_lower = title.lower()

        if not phrase_tokens:
            return 0.0

        matched = sum(1 for t in phrase_tokens if t in title_lower)
        return round((matched / len(phrase_tokens)) * 100, 2)

    def map_bold_phrases_to_sources(
                self,
                bold_phrases,
                query: str,
                documents,
                title_match_threshold=70,
                confidence_threshold=0.5,
                context_threshold=0.5
        ):

        print("Mapping bold phrases to sources using title matching")

        mappings = {}

        for phrase in bold_phrases:
            best_doc = None
            best_title_pct = 0

            for doc in documents:
                title = doc.metadata.get("title", "")
                if not title:
                    continue

                pct = self._title_match_percentage(phrase, title)

                if pct > best_title_pct:
                    best_title_pct = pct
                    best_doc = doc

            print(f"\n🔎 Phrase: '{phrase}' | Best title match: {best_title_pct}%"
                  f" | Title: '{best_doc.metadata.get('title', '') if best_doc else 'N/A'}'")

            if not best_doc or best_title_pct < title_match_threshold:
                print(f"❌ Below threshold ({title_match_threshold}%), skipping")
                continue

            doc_context = self.extract_context_window(best_doc, phrase)

            mappings[phrase] = {
                "source": best_doc.metadata.get("source"),
                "confidence": best_title_pct,
                "context": doc_context,
                "title_match_pct": best_title_pct
            }

            print(
                f"✅ Mapped '{phrase}' → {best_doc.metadata.get('source')} "
                f"(title match: {best_title_pct}%)"
            )

        return mappings


        
import re

# def extract_bold_words(text: str) -> list:
#     """Extract all unique bold words from markdown text."""
#     bold_pattern = r'\*\*(.*?)\*\*'
#     bold_matches = re.findall(bold_pattern, text)
    
#     # Deduplicate while preserving order
#     seen = set()
#     unique_bold = []
#     for word in bold_matches:
#         clean_word = word.strip()
#         if clean_word and clean_word not in seen:
#             seen.add(clean_word)
#             unique_bold.append(clean_word)
    
#     return unique_bold

# def attach_sources_to_bold_text(response_text: str, bold_source_map: dict) -> str:
#     """
#     Attaches source links below bold phrases if available.
#     """
#     updated_text = response_text

#     for phrase, data in bold_source_map.items():
#         source = data.get("source")
#         if not source:
#             continue

#         bold_phrase = f"**{phrase}**"
#         linked_phrase = f"**[{phrase}]({source})**"
#         # linked_phrase = f"**[{phrase}]**({source})"


#         # Avoid double-linking
#         if linked_phrase in updated_text:
#             continue

#         updated_text = updated_text.replace(bold_phrase, linked_phrase, 1)

#     return updated_text

# def attach_sources_to_bold_text(response_text: str, bold_source_map: dict) -> str:
#     """
#     Attach hyperlinks to bold phrases but ensure each URL
#     is used only once in the response.
#     """

#     updated_text = response_text
#     used_urls = set()

#     for phrase, data in bold_source_map.items():
#         source = data.get("source")

#         if not source:
#             continue

#         # Skip if this URL was already used
#         if source in used_urls:
#             continue

#         bold_phrase = f"**{phrase}**"
#         linked_phrase = f"**[{phrase}]({source})**"

#         if bold_phrase in updated_text:
#             updated_text = updated_text.replace(bold_phrase, linked_phrase, 1)
#             used_urls.add(source)

#     return updated_text

def attach_sources_to_bold_text(response_text: str, bold_source_map: dict) -> str:
    updated_text = response_text
    used_urls = set()

    for phrase, data in bold_source_map.items():
        source = data.get("source")

        if not source:
            continue

        if source in used_urls:
            continue

        bold_phrase = f"**{phrase}**"
        linked_phrase = f"**[{phrase}]({source})**"

        if bold_phrase in updated_text:
            updated_text = updated_text.replace(bold_phrase, linked_phrase, 1)
            used_urls.add(source)

    return updated_text

def extract_bold_words(text: str) -> list:
    import re
    bold_pattern = r'\*\*(.*?)\*\*'
    matches = re.findall(bold_pattern, text)

    cleaned = []
    for m in matches:
        word = m.strip()

        # ❌ skip long sentences
        if len(word.split()) > 6:
            continue

        # ❌ skip punctuation-heavy strings
        if len(word) > 60:
            continue

        cleaned.append(word)

    return list(dict.fromkeys(cleaned))


# def map_bold_words_to_sources_with_threshold(
#     bold_words,
#     documents,
#     threshold=80
# ):
#     word_link_map = {}

#     for word in bold_words:
#         word_lower = word.lower().strip()
#         best_doc = None
#         best_score = 0
#         max_possible_score = 0
#         print(f"\n🔎 Trying to map bold word: '{word}'")

#         for doc in documents:
#             title = doc.metadata.get("title", "").lower()
#             content = doc.page_content.lower()
#             print("Mapping Doc Content:", content[:100])  # Print first 100 chars

#             score = 0
#             possible = 0

#             # Title match (strong signal)
#             possible += 3
#             if word_lower in title:
#                 score += 3

#             # Content match (weaker signal)
#             possible += 2
#             if word_lower in content:
#                 score += 2

#             # Normalize to percentage
#             confidence = int((score / possible) * 100) if possible else 0

#             if confidence > best_score:
#                 best_score = confidence
#                 best_doc = doc

#         # Apply threshold
#         if best_doc and best_score >= threshold:
#             word_link_map[word] = {
#                 "link": best_doc.metadata.get("source"),
#                 "confidence": best_score
#             }
#             print(f"Mapped '{word}' to {best_doc.metadata.get('source')} with confidence {best_score}%")

#     return word_link_map




def map_bold_words_to_sources_with_threshold(bold_words, documents, threshold=80):
    word_link_map = {}

    for word in bold_words:
        print(f"\n🔎 Trying to map bold word: '{word}'")
        word_lower = word.lower().strip()

        best_doc = None
        best_confidence = 0

        for doc in documents:
            title = doc.metadata.get("title", "").lower()
            content = doc.page_content.lower()

            score = 0
            possible = 0

            tokens = word_lower.split()

            # title token matches
            possible += len(tokens) * 3
            score += sum(3 for t in tokens if t in title)

            # content token matches
            possible += len(tokens) * 2
            score += sum(2 for t in tokens if t in content)

            confidence = int((score / possible) * 100) if possible else 0

            if confidence > best_confidence:
                best_confidence = confidence
                best_doc = doc

        if best_doc and best_confidence >= threshold:
            word_link_map[word] = {
                "link": best_doc.metadata.get("source"),
                "confidence": best_confidence
            }
            print(
                f"✅ Mapped '{word}' → {best_doc.metadata.get('source')} "
                f"({best_confidence}%)"
            )
        else:
            print(f"confidence", best_confidence)
            print(f"❌ No strong match for '{word}'")
            

    return word_link_map


def hyperlink_bold_words(text, word_link_map):
        for word, data in word_link_map.items():
            link = data["link"]
            bold_pattern = f"**{word}**"
            hyperlink = f"**[{word}]({link})**"
            text = text.replace(bold_pattern, hyperlink)
        return text

class Chatbot:
    def __init__(self):
        self.llm_nostream = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0,
            streaming=False
        )
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0,streaming=True)
        self.llm1 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        self.memory = MemorySaver()
        self.rag_tool = RAGTool()
        self.tool_node = None
        self.app = None
        current_date = datetime.now().strftime("%B %d, %Y")

        # Define the chatbot's system message
        self.system_message = SystemMessage(
            content=(
                "You are IndiaSpend AI, an expert chatbot designed to answer questions related to IndiaSpend articles, reports, and data analysis. "
                "Your responses should be fact-based, sourced ONLY from IndiaSpend's database, and align with IndiaSpend's journalistic style. "
                "IMPORTANT: You must ONLY answer questions using information from IndiaSpend's published articles and data. "
                "Do NOT use your general knowledge to answer questions. If the provided context does not contain relevant information, say so politely. "
                "You should provide clear, well-structured answers and cite sources where applicable. "
                "Website: [IndiaSpend](https://www.indiaspend.com/)."
                "Ensure the response is clear, relevant, and does not mention or imply the existence of any supporting material or Context even if does not help in answering query"
                f"Format: \n**Article Title Mentioned in Article Should be added dynalically:**\nYour summary here\n\n[Read more](Original article URL here)\n"
                f"Note todays date is  {current_date}"
            )
        )

    def setup_tools(self):
        rag_tool = StructuredTool.from_function(
            func=self.rag_tool.retrieve,
            name="RAG",
            description="Retrieve relevant content from IndiaSpend's knowledge base",
            args_schema=RAGQuery
        )
        self.tool_node = ToolNode(tools=[rag_tool])
        
    

    # def should_use_rag(self, query: str) -> bool:
    #     QUESTION_PREFIXES = """[
    # "what actions are needed to",
    # "how is",
    # "what is",
    # "how will",
    # "what measures are in place for",
    # "how many",
    # "what steps are being taken to",
    # "how can",
    # "what challenges do",
    # "how often is",
    # "what are the",
    # "why are",
    # "what sectors are",
    # "how does"]"""
    #     decision_prompt = f"""Determine if external information retrieval needed. Answer yes if query:
    #     - Requires specific facts/data
    #     - References recent events
    #     - Needs domain-specific knowledge
    #     - Requires citations/sources
    #     - if query starts with any of those {QUESTION_PREFIXES} and query makes sense.
    #     - Mostly answer yes looking if it is not satisfying conditions of no
    #     - If it is asking very basic query or topic

    #     Answer no if query:
    #     - If query is invalid like if user typed anything which is senseless for eg: "bcjbsabfshds"
    #     - If query is just hii, hello type of message

    #     Query: {query}"""
    #     # print(decision_prompt)
    #     decision = self.llm.invoke([self.system_message, HumanMessage(content=decision_prompt)])
    #     # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    #     # print(decision.content.lower())
    #     # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    #     return "yes" in decision.content.lower()
    def should_use_rag(self, query: str) -> bool:
        # List of basic greetings to check against
        BASIC_GREETINGS = [
            "hi", "hello", "hey", "hii", "hiii", "hiiii", "helo", 
            "morning", "good morning", "evening", "good evening",
            "afternoon", "good afternoon", "sup", "yo", "hola",
            "greetings", "namaste", "hyy", "hyyy"
        ]
        
        # Clean the query - remove extra spaces and convert to lowercase
        cleaned_query = query.lower().strip()
        
        # Return False only for basic greetings, True for everything else
        if cleaned_query in BASIC_GREETINGS:
            return False
        
        return True
    
    
    # def generate_similar_questions(self, original_query: str, answer: str) -> list[str]:
    #     prompt = f"""
    # You are IndiaSpend AI.

    # Based on the user's question and the answer provided,
    # generate related follow-up questions that help the reader
    # explore the topic more deeply.

    # RULES (STRICT):
    # - Generate exactly 4 questions.
    # - Questions must be directly related to the topic.
    # - Do NOT repeat the original question.
    # - Focus on:
    # • impact
    # • causes
    # • policy or governance
    # • data, trends, or geography
    # - Keep questions short and clear.
    # - Do NOT answer the questions.
    # - Do NOT include explanations.
    # - Do NOT use numbering or bullet points.
    # - Each question must be one sentence.

    # User question:
    # {original_query}

    # Answer:
    # {answer}

    # Return only the questions, one per line.
    # """

    #     response = self.llm_nostream.invoke([HumanMessage(content=prompt)])

    #     questions = [
    #         q.strip()
    #         for q in response.content.split("\n")
    #         if q.strip()
    #     ]

    #     return questions[:4]
    
    def generate_similar_questions(self, original_query: str, answer: str, documents) -> list[str]:

        # Collect article titles
        titles = []
        content_snippets = []

        for doc in documents[:3]:  # use top 3 docs
            title = doc.metadata.get("title", "")
            titles.append(title)

            snippet = doc.page_content[:400].lower()
            content_snippets.append(snippet)

        titles_text = "\n".join(titles)
        content_text = "\n".join(content_snippets)

        prompt = f"""
    You are IndiaSpend AI.

    Generate follow-up questions that help readers explore the topic deeper.

    TOPIC SIGNALS:
    Article Titles:
    {titles_text}

    Article Content:
    {content_text}

    Original Question:
    {original_query}

    Answer:
    {answer}

    STRICT RULES:
    - Generate exactly 4 questions.
    - Questions must be directly related to the article topic.
    - Use signals from titles and article content.
    - Do NOT repeat the original question.
    - Focus on:
    • impact
    • causes
    • policy or governance
    • data, trends, or geography
    - Keep questions short and clear.
    - One sentence per question.
    - Do NOT use numbering or bullet points.
    - Do NOT include explanations.

    Return only the questions, one per line.
    """

        response = self.llm_nostream.invoke([HumanMessage(content=prompt)])

        questions = [
            q.strip()
            for q in response.content.split("\n")
            if q.strip()
        ]

        return questions[:4]


    
    def _handle_greeting(self, query: str) -> dict:
        """Handle greeting/non-RAG queries."""
        print(f"Non-RAG query detected: '{query}'. Generating response without retrieval.")
        greeting_prompt = f"""
        You are IndiaSpend AI.

        The user has sent a greeting.
        Respond politely, briefly, and professionally.
        Introduce yourself in one sentence.
        Ask how you can help with India-related data or stories.
        """

        response = self.llm_nostream.invoke(
            [self.system_message, HumanMessage(content=greeting_prompt)],
            config={"tags": ["final"]}
        )

        print("Generated greeting response:", response.content)
        return {
            "messages": [AIMessage(content=response.content)],
            "sources": [],
            "bold_words": [],
            "similar_questions": []
        }

    def _handle_not_relevant(self, query: str, retrieved_docs=None) -> dict:
        """Handle queries with no relevant articles found."""
        print(f"❌ Query '{query}' has no relevant articles in our database")

        # Find titles from retrieved docs that partially match the query context
        related_articles = []
        if retrieved_docs:
            for doc in retrieved_docs:
                title = doc.metadata.get("title", "")
                source = doc.metadata.get("source", "")
                if not title or not source:
                    continue

                sim = self.rag_tool.title_similarity(query, title)
                pct = round(sim * 100, 2)

                print(f"📰 Title: {title} | Match: {pct}% | Source: {source}")

                if pct >= 40:  # threshold for "related" articles
                    related_articles.append({
                        "title": title,
                        "source": source,
                        "match_pct": pct
                    })

            # Sort by match percentage descending, keep top 3
            related_articles.sort(key=lambda x: x["match_pct"], reverse=True)
            related_articles = related_articles[:3]

        # Build related articles text for the LLM prompt
        related_text = ""
        if related_articles:
            related_text = "\n".join(
                f"- {a['title']} ({a['match_pct']}% match) → {a['source']}"
                for a in related_articles
            )
            print(f"\n✅ Related articles found:\n{related_text}")
        else:
            print("❌ No related articles found either")

        prompt = f"""
        You are IndiaSpend AI.

        The user asked a question, but IndiaSpend's database does not have
        a direct answer for this specific topic.

        User's question:
        {query}

        {"RELATED ARTICLES (suggest these to the user):" + chr(10) + related_text if related_articles else ""}

        RESPONSE RULES (STRICT):
        - Acknowledge the user's topic naturally in 1 sentence.
        - Politely say that IndiaSpend does not have specific coverage on this exact topic.
        - Do NOT make up any facts or data.
        - Do NOT use bold words.
        {"- Suggest the related articles listed above as helpful reads. Format each as: [Article Title](URL)" if related_articles else "- Suggest 2-3 related topics that IndiaSpend typically covers (health, education, environment, gender, governance, economy)."}
        - Keep the response short (3-5 sentences max).
        - Maintain a helpful, journalistic tone.
        """

        response = self.llm_nostream.invoke(
            [self.system_message, HumanMessage(content=prompt)]
        )

        print("Not relevant response:", response.content)

        return {
            "messages": [AIMessage(content=response.content)],
            "sources": [],
            "bold_words": [],
            "similar_questions": []
        }

    def _handle_rag_response(self, query: str, rag_result: dict) -> dict:
        """Handle normal RAG response with relevant articles."""
        sources = rag_result['sources'][:5]
        retrieved_docs = rag_result.get("docs", [])

        date_news = None
        best_source_url = None

        if retrieved_docs:
            best_doc = retrieved_docs[0]

            print("\n================ BEST DOCUMENT METADATA ================\n")
            print(best_doc.metadata)
            print("\n========================================================\n")

            best_source_url = best_doc.metadata.get("source")

            raw_date = best_doc.metadata.get("date_news")
            if raw_date:
                try:
                    date_news = datetime.strptime(
                        raw_date, "%Y-%m-%d %H:%M:%S.%f"
                    ).strftime("%B %d, %Y")
                except Exception:
                    date_news = raw_date.split(" ")[0]

        sources = sources[:5]
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print(sources)
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        # Build context from actual document content (not just URLs)
        doc_context = ""
        for i, doc in enumerate(retrieved_docs[:5]):
            title = doc.metadata.get("title", "")
            content = doc.page_content[:800]
            doc_context += f"Article {i+1}: {title}\n{content}\n\n"

        prompt = f"""
            You are IndiaSpend AI.

            Answer the question below in a clean, reader-friendly format.
            DO NOT use section headings or labels.
            IMPORTANT: Answer ONLY using the information provided in the context below.
            Do NOT use any outside knowledge. If the context does not contain enough information to answer the question, say so.

            Question:
            {query}

            Context (internal use only — from IndiaSpend articles):
            {doc_context}

            Source URLs (internal use only):
            {sources}

            RESPONSE RULES (STRICT):
            - Start with a short paragraph of 2–3 sentences summarising the answer, Bold as many important words as needed to improve clarity and emphasis, but do not overuse.
            - After the paragraph, list 3–5 bullet points only.
            - Bullet points must:
                - start with "•"
                - be sorted by importance
                - include numbers or data where possible
            - Do NOT write labels like "Short Summary", "Key Points", or similar.
            - Do NOT mention context, sources, or retrieval.
            - Maintain a factual, journalistic tone.
            - Base your answer STRICTLY on the provided context. Do NOT add information from outside the context.

            ALLOWED FORMATTING:
            - Use "**bold**" to highlight:
                **years and timelines**
                **key outcomes or results**
                **policy names or programs**
                **important names (people, organizations, places)**
                **critical terms or concepts**
                **major actions or decisions**
                **trends or comparisons**
            - Bold as many important words as needed to improve clarity and emphasis.
            - Bold words are work like **[]()**.
            - Use "•" for bullet points



            STRICT OUTPUT RULES:
            - Do NOT use square brackets [ ] or parentheses ( ).
            - Do NOT include any URLs or links.
            - Do NOT mention sources, references, or context.
            - Do NOT use headings or labels.

            End with:
            [Read more](Original article URL here)\n"
            """

        response = self.llm_nostream.invoke(
            [self.system_message, HumanMessage(content=prompt)]
        )

        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print(response.content)
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

        # Detect if LLM itself says it can't answer (context not relevant)
        not_relevant_phrases = [
            "does not contain specific information",
            "couldn't find any relevant",
            "no relevant information",
            "does not contain enough information",
            "not contain relevant information",
            "sorry, but the provided context",
            "i don't have specific information",
            "no specific information",
        ]
        response_lower = response.content.lower()
        if any(phrase in response_lower for phrase in not_relevant_phrases):
            print("⚠️ LLM response indicates no relevant data — falling back to _handle_not_relevant")
            return self._handle_not_relevant(query, retrieved_docs)

        # Extract bold words from response
        bold_words = extract_bold_words(response.content)

        print("🔍 Extracted Bold Words:")
        print(bold_words)
        print("=" * 80)

        bold_source_map = self.rag_tool.map_bold_phrases_to_sources(
            bold_phrases=bold_words,
            query=query,
            documents=retrieved_docs,
            confidence_threshold=0.5,
            context_threshold=0.5
        )

        # -------- Remove duplicate URLs --------
        unique_source_map = {}
        used_urls = set()

        for phrase, data in bold_source_map.items():
            url = data.get("source")
            if not url:
                continue
            if url in used_urls:
                continue
            unique_source_map[phrase] = data
            used_urls.add(url)

        bold_source_map = unique_source_map

        # Build bold → URL instructions
        bold_url_instructions = []
        for phrase, data in bold_source_map.items():
            if data.get("source"):
                bold_url_instructions.append(
                    f"- {phrase} → {data['source']}"
                )
        bold_url_block = "\n".join(bold_url_instructions)

        # Prepare disclaimer
        disclaimer_text = ""
        print("Metadata date_news:", date_news)
        if date_news:
            disclaimer_text = f"Disclaimer: This news was originally published on {date_news}\n\n"

        # Final formatting pass
        final_prompt = f"""
            You are IndiaSpend AI.

            You are given an article draft and a list of bold phrases with their URLs.

            IMPORTANT TASK:
            Convert bold phrases into hyperlinks ONLY if the URL has not already been used earlier in the text.

            IMPORTANT:
            Add this line at the before the read more of the response:

            *{disclaimer_text.strip()}*

            HYPERLINK FORMAT:
            **phrase** → **[phrase](URL)**

            CRITICAL RULES (STRICT):
            1. Each URL can be used ONLY ONE TIME in the entire response.
            2. If a URL has already been used earlier in the text, DO NOT link it again.
            3. Leave the bold phrase as **phrase** if its URL was already used.
            4. Do NOT invent links.
            5. Do NOT change the wording.
            6. Do NOT remove or add bold phrases.
            7. Preserve formatting exactly.
            8. Do NOT wrap the response in code blocks.
            9. Do NOT use ``` or ```markdown.

            PROCESS:
            - Read the text from top to bottom.
            - When you hyperlink a phrase, mark that URL as used.
            - If you see another phrase mapped to the same URL later, leave it as plain bold text.

            BOLD PHRASE → URL MAP:
            {bold_url_block}

            TEXT:
            {response.content}

            Return ONLY the updated text.
            """

        final_response = self.llm.invoke(
            [HumanMessage(content=final_prompt)],
            config={"tags": ["final"]}
        )

        for k, v in bold_source_map.items():
            print(f"\n🔗 {k}")
            print("Source:", v["source"])
            print("Title Match:", f"{v.get('title_match_pct', v.get('confidence'))}%")
            print("Context:", v["context"])

        response_with_sources = final_response.content

        # Add disclaimer at top using metadata date_news
        if date_news:
            disclaimer = f"Disclaimer: This news was originally published on {date_news}.\n\n"
            response_with_sources = disclaimer + response_with_sources

        # Add extra spacing before first bullet
        response_with_sources = response_with_sources.replace("\n•", "\n\n•", 1)

        print("Final Response with Hyperlinked Bold Words:", response_with_sources)

        # Generate similar questions based on final answer
        retrieved_docs = rag_result.get("docs", [])

        similar_questions = self.generate_similar_questions(
            original_query=query,
            answer=response_with_sources,
            documents=retrieved_docs
        )

        print("🔁 Similar Questions:")
        print(similar_questions)

        return {
            "messages": [AIMessage(content=response_with_sources)],
            "sources": sources,
            "bold_words": bold_words,
            "similar_questions": similar_questions
        }

    def call_model(self, state: MessagesState) -> dict:
        messages = state['messages']
        last_message = messages[-1]
        query = last_message.content
        should_use_rag = self.should_use_rag(query)
        print(f"RAG decision for query '{query}': {should_use_rag}")

        if not should_use_rag:
            result = self._handle_greeting(query)
        else:
            processed_query = preprocess_query(query)
            rag_result = self.rag_tool.retrieve(RAGQuery(query=processed_query))

            if not rag_result.get("is_relevant", True):
                result = self._handle_not_relevant(query, rag_result.get("docs", []))
            else:
                result = self._handle_rag_response(query, rag_result)

        print("\n================ FINAL MESSAGE PASSED ================\n")
        print(result["messages"][0].content)
        print("\n=====================================================\n")

        return result

    def router_function(self, state: MessagesState) -> Literal["tools", END]:
        messages = state['messages']
        last_message = messages[-1]
        return "tools" if getattr(last_message, 'tool_calls', None) else END
    
    def generate_clarifying_questions(self, query: str) -> list[str]:
            prompt = f"""
        You are IndiaSpend AI.

        The user has asked a broad or ambiguous question.
        Your task is to ask follow-up questions that help narrow the scope
        by staying CLOSE to the topic of the original question.

        IMPORTANT:
        - Questions must be directly related to the subject of the query.
        - Ask like a journalist clarifying an interview question.
        - Avoid generic questions like "what topic" or "which sector".
        - Make the questions sound natural and human.

        RULES (STRICT):
        - Generate 1 to 3 clarification questions only.
        - Each question must help narrow:
        • geography (national vs state vs city)
        • scope (public vs private, policy vs impact, causes vs effects)
        • timeframe (recent vs long-term), if relevant
        - Do NOT answer the original question.
        - Do NOT explain why you are asking.
        - Do NOT use bullet points or numbering.
        - Each question must be a single sentence.
        - Keep language simple and conversational.

        User question:
        {query}

        Return only the questions, each on a new line.
        """

            response = self.llm1.invoke([HumanMessage(content=prompt)])

            questions = [
                line.strip()
                for line in response.content.split("\n")
                if line.strip()
            ]

            return questions[:3]
    
    def generate_related_questions(self, query: str, answer: str) -> list[str]:
        prompt = f"""
    You are IndiaSpend AI.

    Based on the user's question and the answer provided,
    generate related follow-up questions that help the reader
    explore the topic more deeply.

    RULES (STRICT):
    - Generate exactly 4 questions.
    - Questions must be directly related to the topic.
    - Do NOT repeat the original question.
    - Focus on:
    • impact
    • causes
    • policy or governance
    • data, trends, or geography
    - Keep questions short and clear.
    - Do NOT answer the questions.
    - Do NOT include explanations.
    - Do NOT use numbering or bullet points.
    - Each question must be one sentence.

    User question:
    {query}

    Answer:
    {answer}

    Return only the questions, one per line.
    """
        response = self.fast_llm.invoke([HumanMessage(content=prompt)])

        questions = [
            line.strip()
            for line in response.content.split("\n")
            if line.strip()
        ]

        return questions[:4]

        
    
    def __call__(self):
        self.setup_tools()
        workflow = StateGraph(MessagesState)
        
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", self.tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            self.router_function,
            {"tools": "tools", END: END}
        )
        # workflow.add_edge("tools", "agent")
        
        self.app = workflow.compile(checkpointer=self.memory)
        return self.app
    
    # def __call__(self):
    #     workflow = StateGraph(MessagesState)

    #     workflow.add_node("agent", self.call_model)
    #     workflow.add_edge(START, "agent")
    #     workflow.add_edge("agent", END)

    #     self.app = workflow.compile(checkpointer=self.memory)
    #     return self.app

