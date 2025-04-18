import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define directories where your documents are stored
documents_dir = "./documents"
pdf_dir = os.path.join(documents_dir, "pdfs")
# text_dir = os.path.join(documents_dir, "texts")

# Create directories if they don't exist
os.makedirs(pdf_dir, exist_ok=True)
# os.makedirs(text_dir, exist_ok=True)

# Step 1: Load documents from directories
def load_documents():
    # Load PDFs
    pdf_loader = DirectoryLoader(
        pdf_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    
    # # Load text files
    # text_loader = DirectoryLoader(
    #     text_dir,
    #     glob="**/*.txt",
    #     loader_cls=TextLoader
    # )
    
    # Load all documents
    pdf_documents = pdf_loader.load()
    # text_documents = text_loader.load()
    
    all_documents = pdf_documents #+ text_documents
    print(f"Loaded {len(all_documents)} documents in total")
    return all_documents

# Step 2: Split documents into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    document_chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(document_chunks)} chunks")
    return document_chunks

# Step 3: Create embeddings and store in ChromaDB
def create_vector_store(document_chunks):
    if not document_chunks:
        print("No document chunks to process")
        return None
    
    # Initialize embeddings provider
    embeddings = OpenAIEmbeddings()
    
    # Print first chunk to verify content
    print(f"Sample chunk content: {document_chunks[0].page_content[:100]}...")
    
    try:
        # Test embedding a single text
        test_embedding = embeddings.embed_query("Test query")
        print(f"Test embedding successful, dimensions: {len(test_embedding)}")
        
        # Create a Chroma vector store from documents
        print("Creating vector store...")
        vector_store = Chroma.from_documents(
            documents=document_chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        # Persist the database
        vector_store.persist()
        print("Documents successfully ingested into ChromaDB")
        return vector_store
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        print("Hint: Make sure your OPENAI_API_KEY is correct and has sufficient credits")
        return None

def main():
    # Load all documents
    documents = load_documents()
    
    # Split into chunks
    document_chunks = split_documents(documents)
    
    if not document_chunks:
        print("No document chunks to process. Exiting.")
        return
    
    # Create and persist vector store
    vector_store = create_vector_store(document_chunks)
    
    if vector_store:
        # Test a simple query
        query = "What are the main topics covered in these documents?"
        results = vector_store.similarity_search(query, k=3)
        
        print("\nTest query results:")
        for i, doc in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Content: {doc.page_content[:200]}...")

if __name__ == "__main__":
    print('cheikh')
    main()