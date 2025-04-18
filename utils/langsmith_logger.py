from langsmith import Client
import config

def log_to_langsmith_dataset(user_input, ai_output, dataset_name="Chatbot Evaluation Dataset"):
    client = Client(api_key=config.LANGCHAIN_API_KEY)

    dataset = client.read_dataset(dataset_name=dataset_name)
    if not dataset:
        dataset = client.create_dataset(dataset_name=dataset_name)

    client.create_example(
        inputs={"input": user_input},
        outputs={"output": ai_output},
        dataset_id=dataset.id
    )
