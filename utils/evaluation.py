from langsmith import Client
from langchain.smith import RunEvalConfig
import config

def setup_langsmith_evaluation():
    """Set up LangSmith for evaluation."""
    client = Client(api_key=config.LANGCHAIN_API_KEY)
    
    # Set up evaluation config
    eval_config = RunEvalConfig(
        evaluators=[
            "qa",  # Question-Answering evaluator
            "criteria", # Custom criteria evaluator
        ],
        custom_evaluators=[]
    )
    
    return client, eval_config

def run_benchmark_evaluation(client, eval_config, benchmark_dataset_name):
    """Run evaluation on a benchmark dataset."""
    # Get dataset
    dataset = client.read_dataset(dataset_name=benchmark_dataset_name)
    
    # Run evaluation
    evaluation_results = client.run_on_dataset(
        dataset_name=benchmark_dataset_name,
        llm_or_chain_factory=None,  # Will be set in main.py
        evaluation=eval_config,
        project_name=config.LANGSMITH_PROJECT,
        concurrency_level=5,
        verbose=True
    )
    
    return evaluation_results