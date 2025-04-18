from typing import Dict, List, Any
from langchain_google_genai import ChatGoogleGenerativeAI
import config

def create_chromadb_agent(vectorstore):
    """Create an agent that retrieves information from ChromaDB."""
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        temperature=0.2,
        google_api_key=config.GOOGLE_API_KEY,
    )
    
    def chromadb_response(state):
        """Generate a response using ChromaDB retrieval."""
        messages = state.messages
        
        # Get the last message from the user
        last_message = None
        for message in reversed(messages):
            if message["role"] == "human":
                last_message = message
                break
        
        if not last_message:
            return {"messages": messages}
        
        # Retrieve relevant documents
        query = last_message["content"]
        docs = vectorstore.similarity_search(query, k=5)
        
        # Format documents into context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Format prompt with context
        system_prompt = f"""You are a helpful AI assistant that provides accurate information based on the retrieved documents.
        Use the context provided to answer the user's question thoroughly.
        If the context doesn't contain the answer, say that you don't have enough information rather than making up an answer.
        
        Context:
        {context}
        
        Answer the following question based on the context above:
        """
        
        prompt = f"{system_prompt}\n\nQuestion: {query}\n\nAnswer:"
        
        # Generate response
        answer = llm.invoke(prompt) #*replace .invoke with .predict
        
        # Extract documents for citation
        sources = []
        if docs:
            for doc in docs:
                if hasattr(doc, "metadata") and doc.metadata:
                    source = doc.metadata.get("source", "Unknown document")
                    if source not in sources:
                        sources.append(source)
        
        # Update state with AI response
        updated_messages = messages.copy()
        updated_messages.append({
            "role": "ai",
            "content": answer,
            "source": "vectorstore_rag",
            "citations": sources
        })
        
        return {"messages": updated_messages}
    
    return chromadb_response

