import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "multi-source-rag")
LANGSMITH_DATASET_NAME = os.getenv("LANGSMITH_DATASET_NAME", "Chatbot Evaluation Dataset")

# Model Configuration
LLM_MODEL = "gemini-2.0-flash"  # Free version of Gemma 
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # Open-source embedding model

# Vector Store Configuration
CHROMA_PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "document_collection"

# Web Search Configuration
SEARCH_ENGINE = "tavily"  # Options: "tavily", "serper"

# Document Settings
DOCUMENT_DIR = "documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Router Thresholds
CERTAINTY_THRESHOLD = 0.7
RECENCY_THRESHOLD = 0.6