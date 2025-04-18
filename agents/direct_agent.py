from typing import Dict, List, Any
from langchain_google_genai import ChatGoogleGenerativeAI
import config

def create_direct_llm_agent():
    """Create an agent that responds directly using the LLM."""
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,  
        temperature=0,
        google_api_key=config.GOOGLE_API_KEY, 
    )
    
    system_prompt = """You are a helpful AI assistant that provides accurate and informative responses.
    Answer the user's question to the best of your knowledge and abilities.
    If you don't know something, admit it rather than making up information.
    """
    
    def direct_llm_response(state):
        """Generate a direct response from the LLM."""
        messages = state.messages
        
        # Format complete conversation for LLM
        formatted_conversation = system_prompt + "\n\n"
        
        for message in messages:
            if message.get("type") == "human":
                formatted_conversation += f"User: {message['content']}\n"
            elif message.get("type") == "ai":
                formatted_conversation += f"Assistant: {message['content']}\n"
        
        # Add prompt for next response
        formatted_conversation += "Assistant: "
        
        # Generate response
        response = llm.invoke(formatted_conversation)                       #*replace .predict with .invoke 
        
        # Update state with AI response
        updated_messages = messages.copy()
        updated_messages.append({
            "role": "ai",
            "content": response,
            "source": "direct_llm"
        })
        
        return {"messages": updated_messages}
    
    return direct_llm_response