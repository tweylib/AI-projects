from utils.evaluation import setup_langsmith_evaluation, run_benchmark_evaluation

client, eval_config = setup_langsmith_evaluation()
run_benchmark_evaluation(client, eval_config, benchmark_dataset_name="Chatbot Evaluation Dataset")
