# run_evaluation.py
import sys
import config
from utils.document_loader import load_documents, process_documents_to_chromadb
from utils.evaluation import setup_langsmith_evaluation, run_benchmark_evaluation
from main import initialize_system  # Import from your main file

def main():
    """Run system in evaluation mode."""
    print("Starting evaluation mode...")
    
    # Check if LangSmith API key is configured
    if not hasattr(config, 'LANGCHAIN_API_KEY') or not config.LANGCHAIN_API_KEY:
        print("Error: LANGCHAIN_API_KEY not set in config. Evaluation cannot proceed.")
        return
    
    if not hasattr(config, 'LANGCHAIN_PROJECT') or not config.LANGCHAIN_PROJECT:
        print("Error: LANGCHAIN_PROJECT not set in config. Evaluation cannot proceed.")
        return
    
    # Initialize system
    workflow = initialize_system()
    
    # Setup evaluation
    client, eval_config = setup_langsmith_evaluation()
    
    # Get benchmark dataset name
    if len(sys.argv) > 1:
        benchmark_dataset_name = sys.argv[1]
    else:
        benchmark_dataset_name = input("Enter your LangSmith benchmark dataset name: ")
    
    # Run evaluation
    try:
        results = run_benchmark_evaluation(
            client=client,
            eval_config=eval_config,
            benchmark_dataset_name=benchmark_dataset_name,
            workflow=workflow
        )
        
        print("\nEvaluation Results:")
        print(f"Total runs: {len(results)}")
        print("Check your LangSmith dashboard for detailed results.")
        print(f"Project: {config.LANGCHAIN_PROJECT}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()