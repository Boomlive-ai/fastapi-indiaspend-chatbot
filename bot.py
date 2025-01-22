from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime

import os
os.environ['OPENAI_API_KEY']= os.getenv("OPENAI_API_KEY")

class RAGQuery(BaseModel):
    query: str = Field(..., description="The query to retrieve relevant content for")

class RAGTool:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.index_name = "india-spend"
        self.vectorstore = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings
        )
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True  # Enable source document return
        )

    def retrieve(self, query: RAGQuery) -> dict:
        # print(f"Retrieving for query: {query.query}")
        similar_docs = self.retriever.get_relevant_documents(query.query)
        source_links = [doc.metadata.get('source', 'No source') for doc in similar_docs]
        
        # result = self.rag_chain.invoke(query.query + "Note do not mention anything about the provided context")
        # if isinstance(result, dict):
        #     response_result = result.get('result', str(result))
        #     source_documents = result.get('source_documents', [])
        #     source_links.extend([doc.metadata.get('source', 'No source') for doc in source_documents])
        # else:
        #     response_result = str(result)
            
        # Remove duplicates while preserving order
        source_links = list(dict.fromkeys(source_links))
        
        return {
            # "result": response_result,
            "sources": source_links
        }

class Chatbot:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        self.memory = MemorySaver()
        self.rag_tool = RAGTool()
        self.tool_node = None
        self.app = None
        current_date = datetime.now().strftime("%B %d, %Y")

        # Define the chatbot's system message
        self.system_message = SystemMessage(
            content=(
                "You are IndiaSpend AI, an expert chatbot designed to answer questions related to IndiaSpend articles, reports, and data analysis. "
                "Your responses should be fact-based, sourced from IndiaSpend's database, and align with IndiaSpend's journalistic style. "
                "You should provide clear, well-structured answers and cite sources where applicable. "
                "Website: [IndiaSpend](https://www.indiaspend.com/)."
                "Ensure the response is clear, relevant, and does not mention or imply the existence of any supporting material or Context even if does not help in answering query"
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
            "greetings", "namaste"
        ]
        
        # Clean the query - remove extra spaces and convert to lowercase
        cleaned_query = query.lower().strip()
        
        # Return False only for basic greetings, True for everything else
        if cleaned_query in BASIC_GREETINGS:
            return False
        
        return True
    def call_model(self, state: MessagesState) -> dict:
        messages = state['messages']
        last_message = messages[-1]
        query = last_message.content
        sources = []
        should_use_rag = self.should_use_rag(query)
        if should_use_rag:
            # print(f"Triggering RAG tool for query: {query}")
            rag_result = self.rag_tool.retrieve(RAGQuery(query=query))
            # result_text = rag_result['result']
            sources = rag_result['sources']
            
            # Format response with context and sources
            context = f"Context: {sources}"
            prompt = (
            f"Answer the following question accurately and with data-backed insights in IndiaSpend's reporting style. "
            f"Ensure the response is clear, relevant, and does not mention or imply the existence of any supporting material or Context:\n\n"
            f"Question: {query}\n\n"
            f"Context: {context}"
            f"Note do not mention that you are using provided context and if context doesnt have anything related to query frame it as directly india spend bot is answering "
            )



            response = self.llm.invoke([self.system_message, HumanMessage(content=prompt)])
            # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            # print(response.content)
            # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

            formatted_response = f"{response.content}\n\nSources:\n" + "\n".join(sources)
            return {"messages": [AIMessage(content=formatted_response)], "sources":sources}
        
        # print([self.system_message] + messages)
        # For non-RAG queries, process normally
        response = self.llm.invoke([self.system_message] + messages)
        # print("##################################################################")
        # print(response.content)
        # print("##################################################################")

        return {"messages": [AIMessage(content=response.content)] , "sources":sources}

    def router_function(self, state: MessagesState) -> Literal["tools", END]:
        messages = state['messages']
        last_message = messages[-1]
        return "tools" if getattr(last_message, 'tool_calls', None) else END

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
        workflow.add_edge("tools", "agent")
        
        self.app = workflow.compile(checkpointer=self.memory)
        return self.app
