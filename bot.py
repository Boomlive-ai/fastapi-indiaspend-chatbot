# importing a necessary library
import os
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore, Pinecone
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults
import pinecone
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI

class RAGQuery(BaseModel):
    query: str = Field(..., description="The query to retrieve relevant content for")
# Define a Document class to simulate the structure
class Document:
    def __init__(self, id: str, metadata: dict, page_content: str):
        self.id = id
        self.metadata = metadata
        self.page_content = page_content



class RAGTool:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.index_name = "india-spend"  # Replace with your actual index name
        self.vectorstore = PineconeVectorStore(index_name=self.index_name, embedding=self.embeddings)
        self.llm = ChatOpenAI(temperature=0, model_name='gpt-4o')
        self.pine_index =  Pinecone(index_name="india-spend", embedding=self.embeddings)
        self.retriever = self.vectorstore.as_retriever()
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever
        )

    def retrieve(self, query: RAGQuery) -> str:
        print(f"Retrieving for query: {query.query}")
        retriever = self.pine_index.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        similar_docs = retriever.get_relevant_documents(query.query)
        # Extract all source links into a list
        source_links = [doc.metadata['source'] for doc in similar_docs]

        print("SIMILAR_LINKS",source_links)
        result = self.rag_chain.invoke(query.query)
        print(f"This is the result generated from vector store {result}")
        response_result = result['result'] if 'result' in result else "No relevant information found."
        return {"result": response_result, "sources": source_links}


class Chatbot:
    def __init__(self):
        self.llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        self.memory = MemorySaver()
        self.rag_tool = RAGTool()

    def call_tool(self):
        # tool = TavilySearchResults(max_results=2)
        rag_tool = StructuredTool.from_function(
            func=self.rag_tool.retrieve,
            name="RAG",
            description="Retrieve relevant content from the knowledge base",
            args_schema=RAGQuery
        )
        tools = [rag_tool]
        self.tool_node = ToolNode(tools=[rag_tool])
        self.llm_with_tool=self.llm.bind_tools(tools)

    def needs_rag_tool(self, query):
        retrieval_indicators = ["information", "details", "data", "explain", "describe", "overview", "summary", "insight"]
        if any(keyword in query.lower() for keyword in retrieval_indicators):
            return True
        if len(query.split()) > 5:  # Consider longer queries as potentially needing retrieval
            return True
        return False
    
    def should_use_rag(self, query):
        decision_prompt = f"Does this query require retrieving external information to answer accurately just return yes or no? Query: {query}"
        decision = self.llm.invoke([HumanMessage(content=decision_prompt)])
        return "yes" in decision.content.lower()
  
    def call_model(self, state: MessagesState):
        messages = state['messages']
        last_message = messages[-1]
        query = last_message.content
        source_links = []
        print(query, self.should_use_rag(query))
        if  self.should_use_rag(query) == True:
            print(f"Triggering RAG tool for query: {last_message.content}")
            rag_result = self.rag_tool.retrieve(RAGQuery(query=last_message.content))
            result_text = rag_result['result']
            source_links = rag_result['sources']
            
            # Format the response with the sources
            formatted_sources = "\n\nSources:\n" + "\n".join(source_links) if source_links else "\n\nNo sources available."
            context_message = AIMessage(content=f"{result_text}{formatted_sources}")
            
            combined_message = HumanMessage(content=f"Provide Detail Explanation on this Question: {last_message.content}, Consider this context for answering the question and do not mention that you are answering using this context in answer even if context or source is not present\n\n{context_message.content}, Note you are chatbot of IndiaSpeend so answer question in similar style and do not mention anything about links provided")
            messages[-1] = combined_message
        
        response = self.llm_with_tool.invoke(messages)
        
        # Format the response with sources
        if hasattr(response, 'additional_kwargs') and 'tool_calls' in response.additional_kwargs:
            sources = [call.name for call in response.additional_kwargs['tool_calls']]
            formatted_response = f"{response.content}\n\nSources used: {', '.join(sources)}"
        else:
            formatted_response = response.content
        
        return {"messages": [AIMessage(content=formatted_response)]}

    
    def router_function(self, state: MessagesState) -> Literal["tools", END]:
        messages = state['messages']
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    def __call__(self):
        self.call_tool()
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", self.tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", self.router_function, {"tools": "tools", END: END})
        workflow.add_edge("tools", 'agent')
        self.app = workflow.compile(checkpointer=self.memory)
        return self.app
        
    