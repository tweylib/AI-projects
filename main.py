import os
import sys
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
import langchain
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from langsmith import Client
from langchain_core.messages import HumanMessage
import config
from utils.document_loader import load_documents, process_documents_to_chromadb
from agents.router import create_router_agent, RouterState
from agents.direct_agent import create_direct_llm_agent
from agents.chromadb_agent import create_chromadb_agent
from agents.web_search_agent import create_web_search_agent

def create_graph(vectorstore):
    """Create the LangGraph for the multi-source RAG system."""
    # Create agents
    router_agent = create_router_agent()
    direct_llm_agent = create_direct_llm_agent()
    chromadb_agent = create_chromadb_agent(vectorstore)
    web_search_agent = create_web_search_agent()
    
    # Define the graph
    graph = StateGraph(RouterState)
    
    # Add nodes
    graph.add_node("router", router_agent)
    graph.add_node("direct_llm", direct_llm_agent)
    graph.add_node("vectorstore_rag", chromadb_agent)
    graph.add_node("web_search", web_search_agent)
    
    # Add edges
    graph.add_conditional_edges(
        "router",
        lambda state: state.next_module,  # Using dot notation for Pydantic model
        {
            "direct_llm": "direct_llm",
            "vectorstore_rag": "vectorstore_rag",
            "web_search": "web_search",
            None: END
        }
    )
    
    # Connect other nodes to END
    graph.add_edge("direct_llm", END)
    graph.add_edge("vectorstore_rag", END)
    graph.add_edge("web_search", END)
    
    # Set entrypoint
    graph.set_entry_point("router")
    
    return graph.compile()

def initialize_system():
    """Initialize the system components."""
    # Load documents
    documents = load_documents()
    
    # Process documents to ChromaDB
    vectorstore = process_documents_to_chromadb(documents)
    
    # Create the graph
    workflow = create_graph(vectorstore)
    
    return workflow

def chat_loop(workflow):
    """Simple command-line chat interface."""
    print("🤖 Multi-Source RAG Chatbot")
    print("Type 'exit' to quit")
    print("-" * 50)
    
    # Initialize state
    state = RouterState(messages=[])
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        
        updated_messages = state.messages.copy()
        updated_messages.append(HumanMessage(content=user_input))
        state.messages = updated_messages

        # Run the workflow
        final_state = workflow.invoke(state)
        
        # Extract AI response  #*condition where changed
        messages = final_state.get("messages", [])
        if messages and messages[-1]["role"] == "ai":
            ai_message = messages[-1]

            # ai_message = final_state.messages[-1]     #*i put this line here to test if it works
            print(f"\nAI ({ai_message['source']}): {ai_message['content']}")
            
            # Display citations if available
            if "citations" in ai_message and ai_message["citations"]:
                print("\nSources:")
                for source in ai_message["citations"]:
                    print(f"- {source}")
        
        # Update state for next iteration
        state = final_state

def main():
    """Main entry point."""
    # Check for API keys
    if not config.GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY not set in environment variables")
        sys.exit(1)
    
    if not config.GOOGLE_PROJECT_ID:
        print("Error: GOOGLE_PROJECT_ID not set in environment variables")
        sys.exit(1)
    
    if not config.TAVILY_API_KEY:
        print("Warning: TAVILY_API_KEY not set. Web search functionality will be limited.")
    
    # Initialize system
    workflow = initialize_system()
    
    # Start chat loop
    chat_loop(workflow)

if __name__ == "__main__":
    main()