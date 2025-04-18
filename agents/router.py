from typing import Dict, List, Tuple, Any, Literal
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI 
import config
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class RouterState(BaseModel):
    """State for the router agent."""
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    next_module: Literal["direct_llm", "vectorstore_rag", "web_search", None] = None

def create_router_agent():
    """Create a router agent that decides which module to use."""
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,  
        temperature=0,
        google_api_key=config.GOOGLE_API_KEY, 
    )
    
    system_prompt = """You are a router agent that decides which module to use for answering user queries.
    
    Analyze the query to determine the most appropriate module:
    
    1. direct_llm: Use for general knowledge questions, opinions, or creative tasks that don't require specific factual information.
    2. vectorstore_rag: Use for queries that likely need information from our internal knowledge base.
    3. web_search: Use for queries about current events, specific facts, or topics that may need up-to-date information.
    
    Output ONLY ONE of these options without any explanation: "direct_llm", "vectorstore_rag", or "web_search".
    """
    
    def route_query(state):
        """Route the query to the appropriate module."""
        messages = state.messages
        if not messages:
            return {"next_module": None}
            
        # Get the last message from the user
        last_message = messages[-1]
        if last_message["role"] != "human":
            return {"next_module": None}
            
        user_query = last_message["content"]
        
        # Format prompt for router LLM
        full_prompt = f"{system_prompt}\n\nQuery: {user_query}\nDecide routing:"
        
        # Get routing decision
        response = llm.invoke(full_prompt)                        #*replace .predict with .invoke
        decision = response.content.strip().lower()           #*testing to add .content
        
        # Validate decision
        valid_options = ["direct_llm", "vectorstore_rag", "web_search"]
        if decision not in valid_options:
            # Default to direct_llm if invalid response
            decision = "direct_llm"
        
        return {"next_module": decision}
    
    return route_query