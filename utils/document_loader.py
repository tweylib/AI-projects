import os
from typing import List, Dict
import glob
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma.vectorstores import Chroma
import config

def load_documents(directory: str = config.DOCUMENT_DIR) -> List:
    """Load documents from the specified directory."""
    documents = []
    
    # Load PDFs
    for pdf_path in glob.glob(os.path.join(directory, "**/*.pdf"), recursive=True):
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    
    # Load text files
    for txt_path in glob.glob(os.path.join(directory, "**/*.txt"), recursive=True):
        loader = TextLoader(txt_path)
        documents.extend(loader.load())
    
    # Load markdown files
    for md_path in glob.glob(os.path.join(directory, "**/*.md"), recursive=True):
        loader = UnstructuredMarkdownLoader(md_path)
        documents.extend(loader.load())
    
    print(f"Loaded {len(documents)} documents from {directory}")
    return documents

def process_documents_to_chromadb(documents: List, force_reload: bool = False) -> Chroma:
    """Process documents and store them in ChromaDB."""
    # Check if ChromaDB already exists
    if os.path.exists(config.CHROMA_PERSIST_DIRECTORY) and not force_reload:
        print(f"Loading existing ChromaDB from {config.CHROMA_PERSIST_DIRECTORY}")
        embedding_function = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        vectorstore = Chroma(
            persist_directory=config.CHROMA_PERSIST_DIRECTORY,
            embedding_function=embedding_function,
            collection_name=config.COLLECTION_NAME
        )
        return vectorstore
    
    # Create a new ChromaDB
    print("Creating new ChromaDB from documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    
    split_documents = text_splitter.split_documents(documents)
    print(f"Split into {len(split_documents)} chunks")
    
    embedding_function = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=split_documents,
        embedding=embedding_function,
        persist_directory=config.CHROMA_PERSIST_DIRECTORY,
        collection_name=config.COLLECTION_NAME
    )
    
    return vectorstore