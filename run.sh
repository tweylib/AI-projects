# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -U pip
pip install -q -r requirements.txt

# Check if documents directory exists
if [ ! -d "documents" ]; then
    echo "Creating documents directory..."
    mkdir -p documents
    echo "Please place your documents in the 'documents' directory."
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Creating .env file template..."
    cat > .env << EOF
GOOGLE_API_KEY=your-google-api-key
GOOGLE_PROJECT_ID=your-google-project-id
TAVILY_API_KEY=your-tavily-api-key
LANGCHAIN_API_KEY=your-langchain-api-key
LANGCHAIN_PROJECT=multi-source-rag
EOF
    echo "Please update the .env file with your API keys."
fis

# Start the application
echo "How would you like to run the application?"
echo "1) Command-line interface"
echo "2) Streamlit web interface"

read -p "Enter your choice (1/2): " choice

case $choice in
    1)
        echo "Starting Streamlit web interface..."
        streamlit run ui/streamlit_app.py
        ;;
    2)
        echo "Starting command-line interface..."
        python main.py
        ;;
    *)
        echo "Invalid choice. Defaulting to command-line interface..."
        python main.py
        ;;
esac