import streamlit as st
import sys
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.langsmith_logger import log_to_langsmith_dataset

import config
from main import initialize_system
from agents.router import RouterState
from langchain_core.messages import HumanMessage
from langsmith import Client

def main():
    st.set_page_config(
        page_title="Multi-Source RAG Chatbot",
        page_icon="🤖",
        layout="wide",
    )
    
    st.title("🤖 Multi-Source RAG Chatbot")
    st.markdown("""
        This chatbot can answer using three different sources:
        - **Direct LLM**: General knowledge from the model (Gemma)
        - **VectorStore RAG**: Internal documents 
        - **Web Search**: Real-time web information
    """)
    
    # Initialize or get session state variables
    if "workflow" not in st.session_state:
        with st.spinner("Initializing system..."):
            st.session_state.workflow = initialize_system()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "human":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                
                # Display source info if available
                if "source" in message:
                    source_type = message["source"]
                    st.caption(f"Source: {source_type}")
                
                # Display citations if available
                if "citations" in message and message["citations"]:
                    with st.expander("View Sources"):
                        for source in message["citations"]:
                            st.write(f"- {source}")
    
    # Chat input
    if prompt := st.chat_input("Ask something..."):
        # Add user message to state as dictionary
        human_msg = {"role": "human", "content": prompt, "type": "human"}
        st.session_state.messages.append(human_msg)
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Prepare state for workflow
        state = RouterState(messages=st.session_state.messages)
        
        # Run the workflow with spinner
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Run the workflow
                final_state = st.session_state.workflow.invoke(state)
                
                # Extract AI response
                if "messages" in final_state and len(final_state["messages"]) > len(st.session_state.messages):
                    ai_message = final_state["messages"][-1]
                    
                    def log_to_langsmith_dataset(user_input, ai_output, dataset_name=config.LANGSMITH_DATASET_NAME):
                        client = Client(api_key=config.LANGCHAIN_API_KEY)

                        # Check if dataset exists or create
                        dataset = client.read_dataset(dataset_name=dataset_name)
                        if not dataset:
                            dataset = client.create_dataset(dataset_name=dataset_name)

                        # Add example
                        client.create_example(
                            inputs={"input": user_input},
                            outputs={"output": ai_output},
                            dataset_id=dataset.id
                        )


                    # Display AI response
                    #print("\nAI Message:", ai_message["source"], "\n")
                    st.write(ai_message["content"].content)
                    
                    # Display source info
                    if "source" in ai_message:
                        source_type = ai_message["source"]
                        st.caption(f"Source: {source_type}")
                    
                    # Display citations if available
                    if "citations" in ai_message and ai_message["citations"]:
                        with st.expander("View Sources"):
                            for source in ai_message["citations"]:
                                st.write(f"- {source}")
                    log_to_langsmith_dataset(prompt, ai_message["content"].content)

if __name__ == "__main__":
    main()
