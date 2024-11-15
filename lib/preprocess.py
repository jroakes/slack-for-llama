"""Preprocess the data and save it to a file."""

from lib.slack_extract import Slack
from datasets import Dataset


def get_user_real_name(slack, user_id):
    for user in slack.users:
        if user['id'] == user_id:
            return user['profile']['real_name']
    return user_id


def extract_human_content(messages, slack):
    extracted_content = []
    for message in messages:
        if 'text' in message and message['text'].strip():
            content = {
                'text': message['text'],
                'user': get_user_real_name(slack, message.get('user', '')),
                'timestamp': message.get('ts', ''),
                'channel': message.get('channel', '')
            }
            extracted_content.append(content)
    return extracted_content


def format_slack_data(data_dir: str = "data", output_file: str = "slack_conversations.txt") -> str:
    slack = Slack(data_dir=data_dir)
    messages = slack.get_messages()
    
    # Extract human-written content
    human_content = extract_human_content(messages, slack)

    # Sort the messages by timestamp
    human_content.sort(key=lambda x: float(x['timestamp']))

    # Format messages as simple text entries
    dataset = []
    for message in human_content:
        # Format the message as simple text
        text = f"{message['user']}: {message['text']}"
        dataset.append(text)

    # Save dataset to file for reference
    with open(output_file, "w", encoding='utf-8') as f_out:
        for text in dataset:
            f_out.write(text + "\n")

    print(f"Training messages saved to {output_file}")
    return dataset


def tokenize_data(dataset: list, model_name: str = "meta-llama/Llama-3.2-3B"):
    """Convert the dataset into a HuggingFace Dataset format"""
    # Convert the list of texts to a Dataset format
    dataset = Dataset.from_dict({"text": dataset})
    
    # Split the dataset
    dataset = dataset.train_test_split(test_size=0.1)
    
    return dataset


def preprocess_data(data_dir: str = "data", model: str = "meta-llama/Llama-3.2-3B"):
    """Preprocess the data for Llama 3.2 fine-tuning"""
    # Get the text data
    dataset = format_slack_data(data_dir, "slack_conversations.txt")
    
    # Convert to HuggingFace Dataset format
    dataset = tokenize_data(dataset, model_name=model)
    
    return dataset