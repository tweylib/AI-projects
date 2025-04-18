from typing import Dict, List, Any
import json
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
import config

class TavilySearchWrapper:
    """Simple wrapper for Tavily search API."""
    
    def __init__(self, api_key=config.TAVILY_API_KEY):
        self.api_key = api_key
        self.api_url = "https://api.tavily.com/search"
    
    def search(self, query, max_results=5):
        """Perform a search using Tavily API."""
        try:
            headers = {
                "content-type": "application/json"
            }
            
            params = {
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic"
            }
            
            response = requests.post(self.api_url, json=params, headers=headers)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            return response.json()
        except Exception as e:
            print(f"Error in Tavily search: {e}")
            return {"results": []}

def create_web_search_agent():
    """Create an agent that retrieves information from web search."""
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,  
        temperature=0,
        google_api_key=config.GOOGLE_API_KEY, 
    )
    
    search_engine = TavilySearchWrapper()
    
    def web_search_response(state):
        """Generate a response using web search."""
        messages = state.messages
        
        # Get the last message from the user
        last_message = None
        for message in reversed(messages):
            if message["role"] == "human":
                last_message = message
                break
        
        if not last_message:
            return {"messages": messages}
        
        # Perform web search
        search_results = search_engine.search(last_message["content"])
        
        # Format search results for prompt
        formatted_results = []
        sources = []
        
        if "results" in search_results:
            for idx, result in enumerate(search_results["results"], 1):
                title = result.get("title", "No title")
                content = result.get("content", "No content")
                url = result.get("url", "No URL")
                
                formatted_results.append(f"{idx}. {title}\nContent: {content}\nURL: {url}\n")
                sources.append(url)
        
        # Create prompt with search results
        system_prompt = f"""You are a helpful AI assistant that provides accurate information based on web search results.
        Use the search results provided to answer the user's question thoroughly and accurately.
        Always cite your sources by including the relevant URLs at the end of your response.
        If the search results don't contain relevant information, acknowledge this and provide the best answer you can with the available information.
        
        Search Results:
        {"\n".join(formatted_results)}
        """
        
        prompt = f"{system_prompt}\n\nQuestion: {last_message['content']}\n\nAnswer:"
        
        # Generate response
        answer = llm.invoke(prompt)                      #*replace .predict with .invoke
        
        # Update state with AI response
        updated_messages = messages.copy()
        updated_messages.append({
            "role": "ai",
            "content": answer,
            "source": "web_search",
            "citations": sources
        })
        
        return {"messages": updated_messages}
    
    return web_search_response

